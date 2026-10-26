"""Type-first clarification and conservative, product-independent suitability ranking."""

import json
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

from app.catalog import import_catalog, load_catalog_chunks
from app.conversation import apply_plan, blank_needs
from app.llm import LLMConfig
from app.selection_policy import assess_suitability, initial_clarification_policy, settle_optional_questions
from app.sessions import empty_state
from app.shopping import NeedsExtractor, ShoppingAssistant


def needs():
    value = blank_needs()
    value.update(sub_categories=["跑步鞋"], max_price=1000, preferences=["夏天通勤", "轻便"])
    value["constraint_quotes"]["max_price"] = "预算1000元"
    return value


def passage(text, section="商品介绍", key="a"):
    return {"chunk_id": key, "section": section,
            "text": "商品编号：p_test\n商品：测试商品。\n" + text}


class SelectionPolicyTests(unittest.TestCase):
    def test_no_type_does_not_guess_shoes_from_commute(self):
        data = needs()
        data["clarification_questions"] = ["请问男鞋还是女鞋？"]
        result = initial_clarification_policy(data, "预算1000元，希望轻便，适合夏天通勤。",
                                              {"brands": []})
        self.assertEqual([], result["sub_categories"])
        self.assertEqual(["你想选购哪类商品？"], result["clarification_questions"])
        self.assertEqual(1000, result["max_price"])
        self.assertEqual(["夏天通勤", "轻便"], result["preferences"])
        self.assertEqual(["跑步鞋"], data["sub_categories"])

    def test_confirmed_type_does_not_block_on_gender_or_size(self):
        data = needs()
        data["clarification_questions"] = ["您想买男鞋还是女鞋？", "您要多大尺码？"]
        result = initial_clarification_policy(data, "预算1000元，买跑步鞋。", {"brands": []})
        self.assertEqual(["跑步鞋"], result["sub_categories"])
        self.assertEqual([], result["clarification_questions"])

    def test_explicit_unsupported_size_is_not_discarded(self):
        data = needs()
        data["unsupported_constraints"] = ["必须42码"]
        data["clarification_questions"] = ["必须42码的条件是否可以调整？"]
        result = initial_clarification_policy(data, "预算1000元，跑步鞋，必须42码。", {"brands": []})
        self.assertEqual(["必须42码"], result["unsupported_constraints"])
        self.assertTrue(result["clarification_questions"])

    def test_reference_ambiguity_question_is_preserved(self):
        data = needs()
        data["clarification_questions"] = ["你指的是哪一款？"]
        self.assertEqual(["你指的是哪一款？"], settle_optional_questions(data)["clarification_questions"])

    def test_budget_only_uses_general_type_question(self):
        data = needs()
        result = initial_clarification_policy(data, "预算1000元", {"brands": []})
        self.assertEqual("", result["keywords"])
        self.assertEqual(["你想选购哪类商品？"], result["clarification_questions"])

    def test_context_type_completion_removes_optional_question(self):
        state = empty_state()
        state["requirements"] = needs()
        state["requirements"]["sub_categories"] = []
        state["requirements"]["clarification_questions"] = ["你想选购哪类商品？"]
        plan = {"new_search": False, "reset_quote": None,
                "updates": {"sub_categories": ["跑步鞋"]},
                "update_quotes": {"sub_categories": "买跑步鞋"},
                "reference_index": None, "reference_quote": None, "reference_action": "none",
                "price_relation": "none", "price_quote": None, "questions": ["请问男鞋还是女鞋？"]}
        result = apply_plan(plan, "买跑步鞋。", state, {"sub_categories": ["跑步鞋"], "brands": []})
        self.assertEqual([], result["needs"]["clarification_questions"])
        self.assertEqual(1000, result["needs"]["max_price"])
        self.assertEqual(["夏天通勤", "轻便"], result["needs"]["preferences"])

    def test_conditional_limit_is_not_rewritten_as_universal_rejection(self):
        text = "不要穿它日常通勤走长路，避免中底过早损耗。"
        result = assess_suitability(needs(), [passage(text)])
        self.assertEqual("caution", result["status"])
        self.assertEqual("不要穿它日常通勤走长路", result["signals"][0]["quote"])
        self.assertNotIn("不适合", result["label"])

    def test_irrelevant_storage_warning_does_not_demote_commuting(self):
        result = assess_suitability(needs(), [passage("不要高温存放；适合日常通勤，轻便透气。")])
        self.assertEqual("supported", result["status"])

    def test_single_negative_review_does_not_determine_global_fit(self):
        result = assess_suitability(needs(), [
            passage("适合日常通勤，轻便透气。"),
            passage("不推荐日常通勤。", section="商品评价", key="review"),
        ])
        self.assertEqual("supported", result["status"])

    def test_missing_evidence_and_double_negation_remain_unknown(self):
        for text in ("资料只有基础规格。", "并不是不适合通勤。"):
            with self.subTest(text=text):
                self.assertEqual("unknown", assess_suitability(needs(), [passage(text)])["status"])

    def test_negated_preference_does_not_become_positive_match(self):
        data = needs()
        data["preferences"] = ["不要厚重"]
        self.assertEqual("unknown", assess_suitability(
            data, [passage("适合喜欢厚重鞋底的用户。")])["status"])

    def test_supported_product_ranks_ahead_of_high_similarity_limited_product(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "input/服饰/data"
            folder.mkdir(parents=True)
            for product_id, description in (
                ("p_limited", "不要穿它日常通勤走长路。"),
                ("p_supported", "适合日常通勤，轻便透气。"),
            ):
                product = {
                    "product_id": product_id, "title": product_id, "brand": "Sample",
                    "category": "服饰", "sub_category": "跑步鞋", "base_price": 500,
                    "skus": [{"sku_id": "s_" + product_id, "properties": {"规格": "标准"}, "price": 500}],
                    "rag_knowledge": {"marketing_description": description, "official_faq": [], "user_reviews": []},
                }
                (folder / (product_id + ".json")).write_text(json.dumps(product, ensure_ascii=False), encoding="utf-8")
            database = root / "catalog.sqlite"
            import_catalog(root / "input", database)
            descriptions = [c for c in load_catalog_chunks(database) if c["section"] == "商品介绍"]
            class Retriever:
                def search(self, query, top_k, product_ids=None):
                    return {"hits": [{**c, "score": .99 if c["product_id"] == "p_limited" else .1}
                                     for c in descriptions if c["product_id"] in product_ids]}
            extractor = Mock()
            extractor.extract.return_value = (needs(), {"requests": 0})
            answerer = Mock()
            answerer.answer.return_value = {
                "status": "insufficient_evidence", "missing_information": ["测试只验证排序"],
                "llm_stats": {"requests": 0}}
            assistant = ShoppingAssistant(
                extractor, answerer, lambda method, online: Retriever(), database, review_mode="rules")
            result = assistant.run("预算1000元，跑步鞋用于夏天通勤。", method="keyword")
            self.assertEqual(["p_supported", "p_limited"], [c["product_id"] for c in result["candidates"]])
            self.assertEqual(.1, result["candidates"][0]["retrieval_score"])
            self.assertEqual("caution", result["candidates"][1]["suitability"]["status"])

    def test_legacy_cached_guess_is_corrected_without_new_model_request(self):
        with TemporaryDirectory() as folder:
            data = needs()
            data["clarification_questions"] = ["男鞋还是女鞋？"]
            client = Mock()
            client.config = LLMConfig("https://example.invalid/v1", "fake", "not-a-key")
            client.stats = {"requests": 0}
            client.chat_json.return_value = data
            extractor = NeedsExtractor(client, Path(folder) / "needs.sqlite")
            query = "预算1000元，希望轻便，适合夏天通勤。"
            context = {"sub_categories": ["跑步鞋"], "brands": []}
            extractor.extract(query, context, online=True)
            import sqlite3
            with closing(sqlite3.connect(extractor.cache_path)) as connection:
                with connection:
                    connection.execute("UPDATE needs SET payload=?", (json.dumps(data),))
            client.chat_json.reset_mock()
            result, stats = extractor.extract(query, context)
            self.assertEqual([], result["sub_categories"])
            self.assertTrue(stats["cache_hit"])
            client.chat_json.assert_not_called()


if __name__ == "__main__":
    unittest.main()
