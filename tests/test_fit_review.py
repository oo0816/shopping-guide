"""Review coverage, citations, scope downgrades, caching and fail-closed integration."""

import copy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

from app.fit_review import FitReviewer, ReviewError, apply_review, guarded_checks, review_answer
from app.llm import LLMConfig, LLMError
from app.shopping import ShoppingAssistant
from app.retrieval import KeywordRetriever
from app.catalog import import_catalog, load_catalog_chunks


def card():
    return {"product_id": "p_one", "title": "样例跑步鞋", "brand": "Sample", "category": "服饰",
            "sub_category": "跑步鞋", "local_path": "sample.json",
            "sku": {"sku_id": "s_one_39", "price": 500, "properties": {"鞋码": "39码"}},
            "retrieval_score": .8}


def hit(text="适合日常通勤。", section="商品介绍", key="description"):
    return {"chunk_id": key, "product_id": "p_one", "mtm": "", "model": "样例跑步鞋",
            "text": "商品编号：p_one\n商品：样例跑步鞋。\n" + text,
            "section": section, "source_url": "", "local_path": "sample.json", "json_pointer": "/"}


def response(requirement="日常通勤", status="supported", quote="适合日常通勤。", scope="product"):
    return {"checks": [{"product_id": "p_one", "requirement": requirement, "status": status,
                        "reason": "资料支持这项需求。", "conditions": [],
                        "citations": [{"chunk_id": "description", "quote": quote, "applicability": scope}]}]}


class FitReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.client = Mock()
        self.client.config = LLMConfig("https://example.invalid/v1", "fake", "fake-secret", max_tokens=4000)
        self.client.stats = {"requests": 0}
        self.client.chat_json.return_value = response()
        self.reviewer = FitReviewer(self.client, self.root / "review.sqlite")
        self.needs = {"preferences": ["日常通勤"]}
        self.tasks = [{"product_id": "p_one", "requirement": "日常通勤"}]

    def test_supported_check_has_same_product_evidence(self):
        result = guarded_checks(response(), self.tasks, [card()], [hit()])
        self.assertEqual("supported", result[0]["status"])
        self.assertEqual("product", result[0]["citations"][0]["applicability"])

    def test_missing_duplicate_or_foreign_requirement_is_rejected(self):
        for data in ({"checks": []}, {"checks": response()["checks"] * 2}):
            with self.assertRaises(ReviewError):
                guarded_checks(data, self.tasks, [card()], [hit()])
        data = response("轻便")
        with self.assertRaises(ReviewError):
            guarded_checks(data, self.tasks, [card()], [hit()])

    def test_invented_quote_and_cross_product_are_rejected(self):
        for data in (response(quote="不存在的摘录"), response()):
            evidence = [hit()]
            if data["checks"][0]["citations"][0]["quote"] != "不存在的摘录":
                evidence[0]["product_id"] = "p_other"
            with self.assertRaises(ReviewError):
                guarded_checks(data, self.tasks, [card()], evidence)

    def test_42_size_weight_cannot_support_39_size(self):
        data = response("轻便", quote="42码重量185g。", scope="selected_sku")
        data["checks"][0]["reason"] = "42码重量185g，轻便。"
        result = guarded_checks(data, [{"product_id": "p_one", "requirement": "轻便"}],
                                [card()], [hit("42码重量185g。")])
        self.assertEqual("unknown", result[0]["status"])
        self.assertEqual("other_sku", result[0]["citations"][0]["applicability"])

    def test_different_explicit_sku_is_not_selected_sku(self):
        evidence = hit("轻便。")
        evidence["sku_id"] = "s_other"
        data = response("轻便", quote="轻便。", scope="selected_sku")
        result = guarded_checks(data, [{"product_id": "p_one", "requirement": "轻便"}], [card()], [evidence])
        self.assertEqual("unknown", result[0]["status"])

    def test_shortened_quote_cannot_hide_same_sentence_size_qualifier(self):
        data = response("轻便", quote="上脚轻便", scope="product")
        checked = guarded_checks(data, [{"product_id": "p_one", "requirement": "轻便"}],
                                 [card()], [hit("42码重量185g，上脚轻便。")])
        self.assertEqual("unknown", checked[0]["status"])
        self.assertEqual("other_sku", checked[0]["citations"][0]["applicability"])

    def test_single_review_is_not_general_support(self):
        result = guarded_checks(response(), self.tasks, [card()], [hit(section="商品评价")])
        self.assertEqual("unknown", result[0]["status"])
        self.assertEqual("individual_review", result[0]["citations"][0]["applicability"])

    def test_explicit_condition_is_preserved_when_model_ignores_it(self):
        data = response(quote="适合短途通勤")
        evidence = hit("适合短途通勤，但不要用于日常通勤走长路。")
        result = guarded_checks(data, self.tasks, [card()], [evidence])
        self.assertEqual("limited", result[0]["status"])
        self.assertTrue(any("走长路" in value for value in result[0]["conditions"]))

    def test_confirmed_short_trip_condition_does_not_become_long_trip_rejection(self):
        data = response("短途通勤", quote="适合短途通勤，不建议长距离通勤。")
        data["checks"][0]["conditions"] = ["短途通勤"]
        tasks = [{"product_id": "p_one", "requirement": "短途通勤"}]
        checked = guarded_checks(data, tasks, [card()], [hit("适合短途通勤，不建议长距离通勤。")])
        self.assertEqual("supported", checked[0]["status"])
        data["checks"][0]["conditions"] = []
        checked = guarded_checks(data, tasks, [card()], [hit("适合短途通勤，不建议长距离通勤。")])
        self.assertEqual("supported", checked[0]["status"])
        self.assertIn("短途通勤", checked[0]["conditions"])
        data = response("夜间通勤", quote="适合白天通勤，不建议夜间通勤。")
        data["checks"][0]["conditions"] = ["夜间通勤"]
        checked = guarded_checks(data, [{"product_id": "p_one", "requirement": "夜间通勤"}],
                                 [card()], [hit("适合白天通勤，不建议夜间通勤。")])
        self.assertEqual("limited", checked[0]["status"])

    def test_invented_conditions_and_unquoted_parameters_are_rejected(self):
        data = response()
        data["checks"][0]["conditions"] = ["必须每天小于1公里"]
        with self.assertRaises(ReviewError):
            guarded_checks(data, self.tasks, [card()], [hit()])
        data = response()
        data["checks"][0]["reason"] = "重量100g，满足需求。"
        with self.assertRaises(ReviewError):
            guarded_checks(data, self.tasks, [card()], [hit()])

    def test_unknown_is_not_treated_as_supported_or_rewritten(self):
        data = response(status="unknown")
        data["checks"][0]["reason"] = "没有足够资料。"
        checked = guarded_checks(data, self.tasks, [card()], [hit()])
        review = {"checks": checked, "status": "reviewed", "method": "test",
                  "cache_hit": False, "llm_stats": {"requests": 0}}
        ranked = apply_review([card()], review)
        self.assertEqual("unknown", ranked[0]["suitability"]["status"])
        answer = review_answer(review, [hit()])
        self.assertEqual("insufficient_evidence", answer["status"])
        self.assertEqual([], answer["claims"])
        self.assertTrue(answer["missing_information"])

    def test_review_cache_is_revalidated_and_reused_without_api(self):
        first = self.reviewer.review(self.needs, [card()], [hit()], online=True)
        self.assertEqual("reviewed", first["status"])
        self.client.chat_json.reset_mock()
        second = self.reviewer.review(self.needs, [card()], [hit()])
        self.assertTrue(second["cache_hit"])
        self.client.chat_json.assert_not_called()
        changed = card()
        changed["sku"]["properties"]["鞋码"] = "42码"
        third = self.reviewer.review(self.needs, [changed], [hit()])
        self.assertEqual("review_disabled", third["status"])

    def test_invalid_review_is_not_cached(self):
        self.client.chat_json.return_value = {"checks": []}
        first = self.reviewer.review(self.needs, [card()], [hit()], online=True)
        self.assertEqual("invalid_review", first["status"])
        second = self.reviewer.review(self.needs, [card()], [hit()])
        self.assertEqual("review_disabled", second["status"])

    def test_offline_cache_miss_and_api_failure_are_distinct(self):
        offline = self.reviewer.review(self.needs, [card()], [hit()])
        self.assertEqual("review_disabled", offline["status"])
        self.client.chat_json.assert_not_called()
        self.client.chat_json.side_effect = LLMError("模拟模型失败")
        failed = self.reviewer.review(self.needs, [card()], [hit()], online=True)
        self.assertEqual("review_failed", failed["status"])

    def test_oversized_review_stops_before_model_call(self):
        result = self.reviewer.review({"preferences": [str(i) for i in range(9)]}, [card()], [hit()], online=True)
        self.assertEqual("invalid_review", result["status"])
        self.client.chat_json.assert_not_called()

    def test_checks_are_batched_without_dropping_requirements(self):
        needs = {"preferences": [f"需求{i}" for i in range(8)]}
        def generated(messages, online=False):
            import json
            payload = json.loads(messages[1]["content"])
            return {"checks": [{"product_id": t["product_id"], "requirement": t["requirement"],
                                "status": "unknown", "reason": "没有对应资料。", "conditions": [], "citations": []}
                               for t in payload["tasks"]]}
        self.client.chat_json.side_effect = generated
        result = self.reviewer.review(needs, [card()], [hit()], online=True)
        self.assertEqual(8, len(result["checks"]))
        self.assertEqual(2, self.client.chat_json.call_count)

    def test_supported_rank_beats_unknown_even_if_similarity_is_lower(self):
        a, b = card(), card()
        b.update(product_id="p_two", retrieval_score=.1)
        review = {"method": "test", "checks": [
            {"product_id": "p_one", "status": "unknown"},
            {"product_id": "p_two", "status": "supported"}]}
        self.assertEqual("p_two", apply_review([a, b], review)[0]["product_id"])

    def test_review_output_limit_is_bounded_and_in_public_profile(self):
        self.assertEqual(4000, self.client.config.profile["max_tokens"])
        with self.assertRaises(LLMError):
            LLMConfig("https://example.invalid", "fake", max_tokens=True)

    def test_semantic_flow_uses_matrix_instead_of_another_writer(self):
        folder = self.root / "input/服饰/data"
        folder.mkdir(parents=True)
        product = {"product_id": "p_one", "title": "样例跑步鞋", "brand": "Sample",
                   "category": "服饰", "sub_category": "跑步鞋", "base_price": 500,
                   "skus": [{"sku_id": "s_one_39", "properties": {"鞋码": "39码"}, "price": 500}],
                   "rag_knowledge": {"marketing_description": "适合日常通勤。", "official_faq": [], "user_reviews": []}}
        import json
        (folder / "p_one.json").write_text(json.dumps(product, ensure_ascii=False), encoding="utf-8")
        database = self.root / "catalog.sqlite"
        import_catalog(self.root / "input", database)
        extractor = Mock()
        extractor.extract.return_value = (
            {"sub_categories": ["跑步鞋"], "keywords": "", "brand": None, "excluded_brands": [],
             "min_price": None, "max_price": 1000, "preferences": ["日常通勤"],
             "clarification_questions": [], "unsupported_constraints": []}, {"requests": 0})
        data = response()
        data["checks"][0]["citations"][0]["chunk_id"] = "catalog::p_one::description"
        self.client.chat_json.return_value = data
        writer = Mock()
        assistant = ShoppingAssistant(
            extractor=extractor, answerer=writer,
            retriever_factory=lambda method, online: KeywordRetriever(load_catalog_chunks(database)),
            db_path=database, reviewer=self.reviewer, review_mode="semantic")
        result = assistant.run("通勤跑步鞋", online=True, method="keyword")
        self.assertEqual("completed", result["status"])
        self.assertEqual("validated_fit_matrix", result["answer"]["generated_from"])
        writer.answer.assert_not_called()
        offline = assistant.run("通勤跑步鞋", method="keyword")
        self.assertTrue(offline["fit_review"]["cache_hit"])
        writer.answer.assert_not_called()
        broken = Mock()
        broken.review.return_value = {"status": "review_failed", "checks": [],
                                      "error": "模拟审核服务失败", "llm_stats": {"requests": 1}}
        assistant.reviewer = broken
        failed = assistant.run("通勤跑步鞋", online=True, method="keyword")
        self.assertEqual("generation_failed", failed["status"])
        self.assertTrue(failed["review_pending"])
        writer.answer.assert_not_called()

    def test_semantic_benchmark_separates_model_and_guard_accuracy(self):
        from evaluations.run_fit_evals import evaluate
        fixture = {"id": "scope", "title": "范围", "requirement": "轻便",
                   "expected_status": "unknown", "cards": [card()],
                   "evidence": [hit("42码重量185g，上脚轻便。")], "data_kind": "synthetic_controlled_evidence"}
        reviewer = Mock()
        reviewer.review.return_value = {"status": "reviewed",
            "checks": [{"status": "unknown", "conditions": []}],
            "model_checks": [{"status": "supported"}]}
        result = evaluate([fixture], reviewer)
        self.assertEqual(0, result["raw_model_status_correct"])
        self.assertEqual(1, result["guarded_status_correct"])
        self.assertEqual(1, result["passed"])


if __name__ == "__main__":
    unittest.main()
