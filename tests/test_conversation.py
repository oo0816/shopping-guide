"""Persistence, condition merging, reference resolution and cross-session isolation."""

import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

from app.catalog import import_catalog, load_catalog_chunks
from app.conversation import (ConversationAssistant, DialogueExtractor, ReferenceClarification,
                              apply_plan, blank_needs)
from app.llm import LLMConfig, LLMError
from app.rag import GroundingError, validate_answer
from app.retrieval import KeywordRetriever
from app.sessions import SessionConflict, SessionStore, empty_state
from app.shopping import OnlineRequired, ShoppingAssistant, ShoppingError, catalog_context

FIRST = "预算1000元，买跑步鞋，夏天通勤，希望轻便。"


def initial_needs():
    data = blank_needs()
    data.update(sub_categories=["跑步鞋"], max_price=1000, preferences=["夏天通勤", "轻便"])
    data["constraint_quotes"]["max_price"] = "预算1000元"
    return data


def plan(**changes):
    result = {"new_search": False, "reset_quote": None, "updates": {}, "update_quotes": {},
              "reference_index": None, "reference_quote": None, "reference_action": "none",
              "price_relation": "none", "price_quote": None, "questions": []}
    result.update(changes)
    return result


class ConversationTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.db = self.root / "catalog.sqlite"
        folder = self.root / "input/服饰/data"
        folder.mkdir(parents=True)
        for product_id, title, subtype, price in (
            ("p_a", "轻便跑步鞋A", "跑步鞋", 900),
            ("p_b", "轻便跑步鞋B", "跑步鞋", 600),
            ("p_c", "轻便跑步鞋C", "跑步鞋", 300),
            ("p_shirt", "夏季衬衫", "衬衫", 100),
        ):
            product = {"product_id": product_id, "title": title, "brand": "Sample",
                       "category": "服饰鞋包", "sub_category": subtype, "base_price": price,
                       "skus": [{"sku_id": "s_" + product_id, "properties": {"规格": "标准"}, "price": price}],
                       "rag_knowledge": {"marketing_description": "轻便透气，适合夏天通勤。",
                                         "official_faq": [], "user_reviews": []}}
            (folder / (product_id + ".json")).write_text(json.dumps(product, ensure_ascii=False), encoding="utf-8")
        import_catalog(self.root / "input", self.db)
        self.context = catalog_context(self.db)
        self.store = SessionStore(self.root / "sessions.sqlite")
        self.extractor = Mock()
        self.extractor.extract.return_value = (initial_needs(), {"cache_hit": True, "requests": 0})
        self.answerer = Mock()
        self.answerer.answer.return_value = {
            "status": "insufficient_evidence", "claims": [], "references": [],
            "missing_information": ["测试不模拟事实推理。"], "llm_stats": {"requests": 0},
            "cache_hit": False}
        self.shopping = ShoppingAssistant(
            review_mode="rules",
            extractor=self.extractor, answerer=self.answerer, db_path=self.db,
            retriever_factory=lambda method, online: KeywordRetriever(load_catalog_chunks(self.db)))
        self.current_plan = plan()
        self.dialogue = Mock()
        self.dialogue.extract.side_effect = lambda query, snapshot, context, online=False: (
            apply_plan(self.current_plan, query, snapshot["state"], context), {"cache_hit": True, "requests": 0})
        self.assistant = ConversationAssistant(self.store, self.shopping, self.dialogue)

    def start(self, session="demo"):
        result = self.assistant.run(session, FIRST, method="keyword")
        self.assertTrue(result["state_saved"])
        self.assertEqual(1, result["turn"])
        return result

    def test_budget_update_preserves_type_and_preferences(self):
        self.start()
        self.current_plan = plan(updates={"max_price": 800}, update_quotes={"max_price": "预算改成800元"})
        result = self.assistant.run("demo", "预算改成800元，其他不变。", method="keyword")
        self.assertEqual(800, result["requirements"]["max_price"])
        self.assertEqual(["夏天通勤", "轻便"], result["requirements"]["preferences"])
        self.assertEqual(["跑步鞋"], result["requirements"]["sub_categories"])
        self.assertTrue(all(card["sku"]["price"] <= 800 for card in result["candidates"]))
        self.assertEqual(2, self.store.load("demo")["revision"])

    def test_second_card_resolves_actual_display_order(self):
        first = self.start()
        expected = first["candidates"][1]["product_id"]
        self.current_plan = plan(reference_index=2, reference_quote="第二双", reference_action="detail")
        result = self.assistant.run("demo", "第二双适合通勤吗？", method="keyword")
        self.assertEqual([expected], [card["product_id"] for card in result["candidates"]])
        self.assertEqual(expected, result["focus_product_id"])
        self.assertIn("夏天通勤", result["effective_query"])
        self.assertIn("直接回答", self.answerer.answer.call_args.args[0])

    def test_cheaper_uses_reference_price_not_invented_discount(self):
        first = self.start()
        target = first["candidates"][1]
        self.current_plan = plan(reference_index=2, reference_quote="第二双", reference_action="alternatives",
                                 price_relation="cheaper", price_quote="便宜一点")
        result = self.assistant.run("demo", "第二双便宜一点。", method="keyword")
        self.assertEqual(target["sku"]["price"] - 0.01, result["relative_price_ceiling"])
        self.assertTrue(result["candidates"])
        self.assertTrue(all(card["sku"]["price"] < target["sku"]["price"] for card in result["candidates"]))
        self.assertNotIn(target["product_id"], [card["product_id"] for card in result["candidates"]])
        self.assertEqual(1000, result["requirements"]["max_price"])
        self.assertEqual(["夏天通勤", "轻便"], self.store.load("demo")["state"]["requirements"]["preferences"])

    def test_no_cheaper_alternative_does_not_relax_conditions(self):
        first = self.start()
        cheapest_index = min(range(len(first["candidates"])), key=lambda i: first["candidates"][i]["sku"]["price"])
        self.current_plan = plan(reference_index=cheapest_index + 1,
                                 reference_quote=f"第{cheapest_index + 1}双",
                                 reference_action="alternatives", price_relation="cheaper", price_quote="便宜一点")
        result = self.assistant.run("demo", f"第{cheapest_index + 1}双便宜一点。", method="keyword")
        self.assertEqual("no_products", result["status"])
        self.assertEqual([], result["candidates"])
        self.assertEqual(1000, result["requirements"]["max_price"])

    def test_restart_restores_session_from_another_store_instance(self):
        self.start()
        restored = ConversationAssistant(SessionStore(self.store.path), self.shopping, self.dialogue)
        self.current_plan = plan(updates={"max_price": 800}, update_quotes={"max_price": "预算800元"})
        result = restored.run("demo", "预算800元", method="keyword")
        self.assertEqual(2, result["turn"])
        self.assertEqual(["夏天通勤", "轻便"], result["requirements"]["preferences"])

    def test_sessions_do_not_share_context(self):
        self.start("first")
        self.current_plan = plan(updates={"max_price": 400}, update_quotes={"max_price": "预算400元"})
        self.assistant.run("first", "预算400元", method="keyword")
        self.start("second")
        self.assertEqual(400, self.store.load("first")["state"]["requirements"]["max_price"])
        self.assertEqual(1000, self.store.load("second")["state"]["requirements"]["max_price"])
        self.assertEqual(1, self.store.load("second")["revision"])

    def test_missing_or_ambiguous_reference_gets_clarification(self):
        self.start()
        self.current_plan = plan(reference_index=4, reference_quote="第四双", reference_action="detail")
        result = self.assistant.run("demo", "第四双怎么样", method="keyword")
        self.assertEqual("needs_clarification", result["status"])
        self.current_plan = plan(reference_quote="它", reference_action="detail")
        result = self.assistant.run("demo", "它怎么样", method="keyword")
        self.assertEqual("needs_clarification", result["status"])

    def test_large_ordinal_does_not_become_first_item(self):
        self.start()
        self.current_plan = plan(reference_quote="第12双", reference_action="detail")
        result = self.assistant.run("demo", "第12双怎么样", method="keyword")
        self.assertEqual("needs_clarification", result["status"])

    def test_followup_without_any_candidates_does_not_call_model(self):
        result = self.assistant.run("new", "第二个怎么样", method="keyword")
        self.assertEqual("needs_clarification", result["status"])
        self.extractor.extract.assert_not_called()
        self.dialogue.extract.assert_not_called()

    def test_new_search_resets_old_preferences_and_relative_options(self):
        self.start()
        self.current_plan = plan(new_search=True, reset_quote="重新开始",
                                 updates={"sub_categories": ["衬衫"], "max_price": 200},
                                 update_quotes={"sub_categories": "买衬衫", "max_price": "预算200元"})
        result = self.assistant.run("demo", "重新开始，买衬衫，预算200元。", method="keyword")
        self.assertEqual(["衬衫"], result["requirements"]["sub_categories"])
        self.assertEqual([], result["requirements"]["preferences"])
        state = self.store.load("demo")["state"]
        self.assertIsNone(state["focus_product"])
        self.assertIsNone(state["search_options"]["price_ceiling"])
        self.assertEqual(["p_shirt"], [card["product_id"] for card in state["last_candidates"]])

    def test_explicit_brand_and_budget_removal_works(self):
        state = empty_state()
        state["requirements"] = initial_needs()
        state["requirements"].update(brand="Sample")
        state["requirements"]["constraint_quotes"]["brand"] = "只看Sample"
        resolved = apply_plan(plan(updates={"brand": None, "max_price": None},
                                   update_quotes={"brand": "品牌不限", "max_price": "不限制预算"}),
                              "品牌不限，不限制预算", state, self.context)
        self.assertIsNone(resolved["needs"]["brand"])
        self.assertIsNone(resolved["needs"]["max_price"])
        self.assertEqual(["夏天通勤", "轻便"], resolved["needs"]["preferences"])

    def test_patch_cannot_silently_delete_or_invent_hard_condition(self):
        state = empty_state()
        state["requirements"] = initial_needs()
        for bad in (
            plan(updates={"max_price": 2000}, update_quotes={"max_price": "预算800元"}),
            plan(updates={"max_price": None}, update_quotes={}),
        ):
            with self.assertRaises(ShoppingError):
                apply_plan(bad, "预算800元", state, self.context)
        self.assertEqual(1000, state["requirements"]["max_price"])

    def test_model_failure_does_not_overwrite_saved_state(self):
        self.start()
        before = self.store.load("demo")
        self.dialogue.extract.side_effect = LLMError("模拟网络失败")
        result = self.assistant.run("demo", "预算800元", method="keyword")
        self.assertEqual("generation_failed", result["status"])
        self.assertFalse(result["state_saved"])
        after = self.store.load("demo")
        self.assertEqual(before, after)

    def test_optimistic_revision_prevents_stale_overwrite(self):
        self.start()
        snapshot = self.store.load("demo")
        self.store.save("demo", 1, snapshot["state"], "另一轮", {"status": "needs_clarification"})
        with self.assertRaises(SessionConflict):
            self.store.save("demo", 1, empty_state(), "过期写入", {"status": "no_products"})
        self.assertEqual(2, self.store.load("demo")["revision"])

    def test_unknown_name_does_not_fall_back_to_unrelated_focus(self):
        self.start()
        state = self.store.load("demo")["state"]
        state["focus_product"] = state["last_candidates"][0]
        with self.assertRaises(ReferenceClarification):
            apply_plan(plan(reference_quote="不存在的牌子", reference_action="detail"),
                       "不存在的牌子怎么样", state, self.context)

    def test_dialogue_cache_depends_on_context_and_never_calls_offline(self):
        self.start()
        client = Mock()
        client.config = LLMConfig("https://example.invalid/v1", "fake", "test-secret")
        client.stats = {"requests": 0}
        client.chat_json.return_value = plan(updates={"max_price": 800}, update_quotes={"max_price": "预算800元"})
        extractor = DialogueExtractor(client, self.root / "dialogue.sqlite")
        snapshot = self.store.load("demo")
        extractor.extract("预算800元", snapshot, self.context, online=True)
        client.chat_json.reset_mock()
        _, stats = extractor.extract("预算800元", snapshot, self.context)
        self.assertTrue(stats["cache_hit"])
        client.chat_json.assert_not_called()
        changed = copy.deepcopy(snapshot)
        changed["state"]["requirements"]["preferences"] = ["越野"]
        with self.assertRaises(OnlineRequired):
            extractor.extract("预算800元", changed, self.context)
        client.chat_json.assert_not_called()

    def test_clear_relative_price_is_explicit(self):
        state = empty_state()
        state["requirements"] = initial_needs()
        state["search_options"] = {"price_ceiling": 599.99, "excluded_product_ids": ["p_b"]}
        resolved = apply_plan(plan(price_relation="clear", price_quote="不用限制更便宜了"),
                              "不用限制更便宜了", state, self.context)
        self.assertIsNone(resolved["options"]["price_ceiling"])
        self.assertEqual([], resolved["options"]["excluded_product_ids"])

    def test_focused_answer_resolves_omitted_id_only_from_same_product_citations(self):
        chunks = load_catalog_chunks(self.db)
        hit = next(row for row in chunks if row["chunk_id"] == "catalog::p_a::description")
        data = {"status": "answered", "claims": [{"text": "资料介绍轻便透气。", "citations": [
            {"chunk_id": hit["chunk_id"], "quote": "轻便透气，适合夏天通勤。"}]}],
            "missing_information": []}
        scope = {"shopping_allowed_products": ["p_a"], "shopping_focus_product": "p_a"}
        checked = validate_answer(copy.deepcopy(data), [hit], scope)
        self.assertTrue(checked["claims"][0]["text"].startswith("p_a："))
        other_scope = {"shopping_allowed_products": ["p_b"], "shopping_focus_product": "p_b"}
        with self.assertRaises(GroundingError):
            validate_answer(data, [hit], other_scope)
        with self.assertRaises(GroundingError):
            validate_answer(data, [hit], {"shopping_allowed_products": ["p_a", "p_b"]})

    def test_empty_search_keeps_last_displayed_cards_for_later_reference(self):
        first = self.start()
        self.current_plan = plan(updates={"max_price": 20}, update_quotes={"max_price": "预算20元"})
        result = self.assistant.run("demo", "预算20元", method="keyword")
        self.assertEqual("no_products", result["status"])
        saved = self.store.load("demo")["state"]
        self.assertEqual([card["product_id"] for card in first["candidates"]],
                         [card["product_id"] for card in saved["last_candidates"]])
        self.assertEqual(20, saved["requirements"]["max_price"])

    def test_history_uses_bound_limit_while_preserving_revision(self):
        self.start()
        for revision in range(1, 8):
            state = self.store.load("demo")["state"]
            self.store.save("demo", revision, state, f"消息{revision}", {"status": "no_products"})
        loaded = self.store.load("demo", history_limit=3)
        self.assertEqual(8, loaded["revision"])
        self.assertEqual([6, 7, 8], [turn["revision"] for turn in loaded["turns"]])


if __name__ == "__main__":
    unittest.main()
