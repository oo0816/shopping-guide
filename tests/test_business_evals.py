"""Prevent evaluation reporting from hiding failures or claiming cache misses as passes."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from evaluations.run_business_evals import (
    CallBudget, EvaluationBudgetExceeded, check_turn, load_cases, review_markdown, summarize,
)


class BusinessEvaluationTests(unittest.TestCase):
    def test_call_budget_stops_before_next_request(self):
        budget = CallBudget(1, 0)
        budget.take("llm")
        with self.assertRaises(EvaluationBudgetExceeded):
            budget.take("llm")
        with self.assertRaises(EvaluationBudgetExceeded):
            budget.take("embedding")
        self.assertEqual({"llm": 1, "embedding": 0}, budget.counts)

    def test_cache_misses_and_budget_blocks_are_not_passes(self):
        rows = [{"category": "test", "outcome": value}
                for value in ("passed", "failed", "not_run_cache", "not_run_budget")]
        metrics = summarize(rows)
        self.assertEqual(2, metrics["executed_cases"])
        self.assertEqual(.5, metrics["execution_coverage"])
        self.assertEqual(.5, metrics["pass_rate_executed"])
        self.assertEqual(.25, metrics["passed_over_all_cases"])

    def test_zero_execution_does_not_report_perfect_accuracy(self):
        metrics = summarize([{"category": "test", "outcome": "not_run_cache"}])
        self.assertIsNone(metrics["pass_rate_executed"])
        self.assertEqual(0, metrics["passed_over_all_cases"])

    def test_missing_status_assertion_or_duplicate_case_ids_is_rejected(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "cases.jsonl"
            bad = {"id": "one", "turns": [{"query": "鞋", "expect": {}}]}
            path.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_cases(path)
            valid = {"id": "one", "turns": [{"query": "鞋", "expect": {"statuses": ["no_products"]}}]}
            path.write_text("\n".join(json.dumps(valid) for _ in range(2)), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_cases(path)

    def test_wrong_budget_and_lost_preference_are_visible_failures(self):
        result = {
            "status": "no_products", "requirements": {"max_price": 2000, "preferences": []},
            "candidates": [], "state_saved": True,
        }
        before = {"revision": 1, "state": {"requirements": {"max_price": 1000}}}
        after = {"revision": 2, "state": {"requirements": result["requirements"]}}
        checks = check_turn(result, {"statuses": ["no_products"], "requirements": {"max_price": 800},
                                    "preserve_fields": ["preferences"]},
                            [], {"preferences": ["通勤"]}, before, after, {}, {})
        failed = {check["name"] for check in checks if not check["passed"]}
        self.assertIn("requirements.max_price", failed)
        self.assertIn("preserve.preferences", failed)

    def test_relative_price_oracle_does_not_assume_first_item_is_more_expensive(self):
        needs = {"sub_categories": ["跑步鞋"], "brand": None, "excluded_brands": [],
                 "min_price": None, "max_price": 1000}
        result = {"status": "no_products", "requirements": needs, "candidates": [], "state_saved": True}
        before = {"revision": 1, "state": {"search_options": {"price_ceiling": None, "excluded_product_ids": []}}}
        after = {"revision": 2, "state": {"requirements": needs}}
        previous = [{"product_id": "reference", "sku": {"price": 600}}]
        catalog = {"s_other": {"product_id": "other", "price_cents": 50000,
                                 "brand": "Sample", "sub_category": "跑步鞋"}}
        expected = {"statuses": ["no_products"], "cheaper_feasibility_from_previous": 1}
        checks = check_turn(result, expected, previous, needs, before, after, {}, catalog)
        self.assertTrue(any(c["name"] == "alternative_availability" and not c["passed"] for c in checks))
        catalog["s_other"]["price_cents"] = 70000
        checks = check_turn(result, expected, previous, needs, before, after, {}, catalog)
        self.assertTrue(all(c["passed"] for c in checks))

    def test_review_export_keeps_failures_and_does_not_claim_semantic_accuracy(self):
        report = {
            "summary": {"total_cases": 1, "passed": 0, "failed": 1, "not_run_cache": 0, "not_run_budget": 0},
            "results": [{"id": "B01", "title": "测试", "outcome": "failed", "turns": [{
                "turn": 1, "query": "预算800元", "result": {"status": "failed", "error": "模拟失败"},
                "checks": [{"name": "budget", "passed": False, "detail": "超过预算"}],
            }]}],
        }
        text = review_markdown(report)
        self.assertIn("超过预算", text)
        self.assertIn("不代表推荐准确率", text)


if __name__ == "__main__":
    unittest.main()
