"""Business-flow tests with fixture products and simulated model output only."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

from app.catalog import import_catalog, load_catalog_chunks, search_products
from app.llm import LLMConfig
from app.rag import GroundedAnswerer, GroundingError, validate_answer
from app.retrieval import KeywordRetriever
from app.shopping import (NeedsExtractor, OnlineRequired, ShoppingAssistant, ShoppingError,
                          catalog_context, validate_needs)


QUERY = "预算300元，想买双夏天通勤穿的鞋，希望轻便一些。"


def needs():
    return {
        "sub_categories": ["跑步鞋"], "keywords": "", "brand": None,
        "excluded_brands": [], "min_price": None, "max_price": 300,
        "preferences": ["夏天通勤", "轻便"], "unsupported_constraints": [],
        "clarification_questions": [],
        "constraint_quotes": {"brand": None, "excluded_brands": None,
                              "min_price": None, "max_price": "预算300元"},
    }


def fixture(product_id, title, sub_category, prices, brand="Sample"):
    return {
        "product_id": product_id, "title": title, "brand": brand,
        "category": "服饰鞋包", "sub_category": sub_category, "base_price": prices[0],
        "skus": [{"sku_id": f"{product_id}_{index}", "properties": {"规格": f"规格{index}"},
                  "price": price} for index, price in enumerate(prices)],
        "rag_knowledge": {"marketing_description": "轻便透气，适合夏天通勤。",
                          "official_faq": [], "user_reviews": []},
    }


class ShoppingTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        folder = self.root / "input/服饰/data"
        folder.mkdir(parents=True)
        for product in (
            fixture("p_shoe", "通勤跑步鞋", "跑步鞋", [150, 450]),
            fixture("p_premium", "轻便跑步鞋", "跑步鞋", [500], "Other"),
            fixture("p_shirt", "夏季衬衫", "衬衫", [100]),
        ):
            (folder / (product["product_id"] + ".json")).write_text(
                json.dumps(product, ensure_ascii=False), encoding="utf-8")
        self.db = self.root / "catalog.sqlite"
        import_catalog(self.root / "input", self.db)
        self.context = catalog_context(self.db)
        self.client = Mock()
        self.client.config = LLMConfig("https://example.invalid/v1", "fake", "fake-secret")
        self.client.stats = {"requests": 0}
        self.client.chat_json.return_value = needs()
        self.extractor = NeedsExtractor(self.client, self.root / "needs.sqlite")

    def assistant(self, extracted=None, answerer=None):
        extractor = Mock()
        extractor.extract.return_value = (extracted or needs(), {"cache_hit": True, "requests": 0})
        return ShoppingAssistant(
            review_mode="rules",
            extractor=extractor, answerer=answerer or Mock(),
            retriever_factory=lambda method, online: KeywordRetriever(load_catalog_chunks(self.db)),
            db_path=self.db)

    def test_budget_must_match_user_quote(self):
        wrong = needs()
        wrong["max_price"] = 500
        with self.assertRaises(ShoppingError):
            validate_needs(wrong, QUERY, self.context)
        wrong = needs()
        wrong["constraint_quotes"]["max_price"] = "别人说预算300元"
        with self.assertRaises(ShoppingError):
            validate_needs(wrong, QUERY, self.context)

    def test_unknown_categories_brands_and_bad_schema_fail(self):
        for field, value in (("sub_categories", ["火箭"]), ("brand", "Invented"),
                             ("max_price", True), ("preferences", "轻便")):
            data = needs()
            data[field] = value
            with self.subTest(field=field), self.assertRaises(ShoppingError):
                validate_needs(data, QUERY, self.context)
        with self.assertRaises(ShoppingError):
            validate_needs({"max_price": 300}, QUERY, self.context)

    def test_missing_product_type_gets_clarification(self):
        data = needs()
        data["sub_categories"] = []
        result = validate_needs(data, QUERY, self.context)
        self.assertEqual(["你想选购哪类商品？"], result["clarification_questions"])

    def test_numeric_unit_budget_is_supported_without_guessing(self):
        data = needs()
        data["max_price"] = 3000
        data["constraint_quotes"]["max_price"] = "预算3千元"
        result = validate_needs(data, "预算3千元，买鞋", self.context)
        self.assertEqual(3000, result["max_price"])

    def test_needs_cache_replays_without_model_call(self):
        self.extractor.extract(QUERY, self.context, online=True)
        self.client.chat_json.reset_mock()
        _, stats = self.extractor.extract(QUERY, self.context)
        self.assertTrue(stats["cache_hit"])
        self.client.chat_json.assert_not_called()

    def test_offline_cache_miss_never_calls_model(self):
        with self.assertRaises(OnlineRequired):
            self.extractor.extract(QUERY, self.context)
        self.client.chat_json.assert_not_called()

    def test_invalid_needs_are_not_cached(self):
        self.client.chat_json.return_value = {"max_price": 1000}
        with self.assertRaises(ShoppingError):
            self.extractor.extract(QUERY, self.context, online=True)
        with self.assertRaises(OnlineRequired):
            self.extractor.extract(QUERY, self.context)

    def test_exact_subcategory_prevents_clothing_from_matching_shoe_category(self):
        result = search_products(sub_category="跑步鞋", max_price=300, db_path=self.db)
        self.assertEqual(["p_shoe"], [item["product_id"] for item in result["items"]])

    def test_no_products_keeps_budget_and_stops_before_recommendation(self):
        data = needs()
        data["max_price"] = 20
        assistant = self.assistant(data)
        result = assistant.run(QUERY, method="keyword")
        self.assertEqual("no_products", result["status"])
        assistant.answerer.answer.assert_not_called()
        self.assertEqual(20, result["requirements"]["max_price"])

    def test_unsupported_hard_condition_stops_before_search(self):
        data = needs()
        data["unsupported_constraints"] = ["必须完全防水"]
        assistant = self.assistant(data)
        result = assistant.run(QUERY, method="keyword")
        self.assertEqual("needs_clarification", result["status"])
        self.assertEqual([], result["candidates"])
        assistant.answerer.answer.assert_not_called()

    def test_excluded_brand_is_not_silently_reintroduced(self):
        data = needs()
        data["excluded_brands"] = ["Sample"]
        result = self.assistant(data).run(QUERY, method="keyword")
        self.assertEqual("no_products", result["status"])

    def test_end_to_end_uses_eligible_sku_and_same_product_references(self):
        client = Mock()
        client.config = self.client.config
        client.stats = {"requests": 0}
        client.chat_json.return_value = {
            "status": "answered",
            "claims": [{"text": "p_shoe 资料介绍其轻便透气，适合夏天通勤。",
                        "citations": [{"chunk_id": "catalog::p_shoe::description",
                                       "quote": "轻便透气，适合夏天通勤。"}]}],
            "missing_information": [],
        }
        answerer = GroundedAnswerer(client, self.root / "answers.sqlite")
        assistant = self.assistant(answerer=answerer)
        first = assistant.run(QUERY, online=True, method="keyword")
        self.assertEqual("completed", first["status"])
        self.assertEqual(["p_shoe"], [item["product_id"] for item in first["candidates"]])
        self.assertEqual(150, first["candidates"][0]["sku"]["price"])
        self.assertEqual("p_shoe", first["answer"]["references"][0]["product_id"])
        messages = client.chat_json.call_args.args[0]
        self.assertNotIn("450.00元", json.dumps(messages, ensure_ascii=False))
        client.chat_json.reset_mock()
        second = assistant.run(QUERY, method="keyword")
        self.assertEqual("completed", second["status"])
        self.assertTrue(second["answer"]["cache_hit"])
        client.chat_json.assert_not_called()

    def test_candidate_product_guards_reject_wrong_or_missing_id_and_prices(self):
        hit = next(c for c in load_catalog_chunks(self.db)
                   if c["chunk_id"] == "catalog::p_shoe::description")
        for text in ("这双鞋很轻便。", "p_shirt 很轻便。", "p_shoe 售价150元。"):
            answer = {"status": "answered", "claims": [{"text": text, "citations": [
                {"chunk_id": hit["chunk_id"], "quote": "轻便透气，适合夏天通勤。"}]}],
                "missing_information": []}
            with self.subTest(text=text), self.assertRaises(GroundingError):
                validate_answer(answer, [hit], {"shopping_allowed_products": ["p_shoe"]})

    def test_shopping_status_is_derived_without_weakening_citation_checks(self):
        hit = next(c for c in load_catalog_chunks(self.db)
                   if c["chunk_id"] == "catalog::p_shoe::description")
        data = {"status": "answered", "claims": [{
            "text": "p_shoe 资料介绍轻便透气。",
            "citations": [{"chunk_id": hit["chunk_id"], "quote": "轻便透气，适合夏天通勤。"}]}],
            "missing_information": ["尺码需要确认。"]}
        scope = {"shopping_allowed_products": ["p_shoe"]}
        checked = validate_answer(data, [hit], scope)
        self.assertEqual("partial", checked["status"])
        data["claims"][0]["citations"][0]["quote"] = "不存在的资料"
        with self.assertRaises(GroundingError):
            validate_answer(data, [hit], scope)

    def test_candidate_explanation_cannot_borrow_other_product_quote(self):
        passages = [c for c in load_catalog_chunks(self.db) if c["section"] == "商品介绍"]
        wrong = next(c for c in passages if c["product_id"] == "p_shirt")
        data = {"status": "answered", "claims": [{
            "text": "p_shoe 轻便透气。",
            "citations": [{"chunk_id": wrong["chunk_id"], "quote": "轻便透气，适合夏天通勤。"}]}],
            "missing_information": []}
        with self.assertRaises(GroundingError):
            validate_answer(data, passages, {"shopping_allowed_products": ["p_shoe", "p_shirt"]})


if __name__ == "__main__":
    unittest.main()
