"""Offline graph tests: real catalog/checkpoints; injected model and retrieval failures."""
import copy
import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from app.catalog import import_catalog, load_catalog_chunks
from app.conversation import blank_needs
from app.retrieval import KeywordRetriever
from app.shopping import OnlineRequired, ShoppingAssistant

try:
    from app.graph_workflow import GraphAssistant, ShoppingWorkflow, thread_lock
except ModuleNotFoundError as exc:
    if not exc.name.startswith("langgraph"):
        raise
    GraphAssistant = None


@unittest.skipIf(GraphAssistant is None, "Install requirements.txt in .venv for graph tests")
class GraphTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        folder = self.root / "input/shoes/data"
        folder.mkdir(parents=True)
        product = {"product_id": "p_shoe", "title": "通勤跑步鞋", "brand": "Sample",
            "category": "服饰鞋包", "sub_category": "跑步鞋", "base_price": 150,
            "skus": [{"sku_id": "s_39", "price": 150, "properties": {"鞋码": "39码"}},
                     {"sku_id": "s_42", "price": 450, "properties": {"鞋码": "42码"}}],
            "rag_knowledge": {"marketing_description": "适合日常通勤。", "official_faq": [], "user_reviews": []}}
        (folder / "shoe.json").write_text(json.dumps(product, ensure_ascii=False), encoding="utf-8")
        self.catalog = self.root / "catalog.sqlite"
        import_catalog(self.root / "input", self.catalog)
        self.needs = blank_needs()
        self.needs.update(sub_categories=["跑步鞋"], max_price=300)
        self.extractor = Mock()
        self.extractor.extract.side_effect = lambda *a, **kw: (copy.deepcopy(self.needs), {"cache_hit": True, "requests": 0})
        self.answerer = Mock()
        self.answerer.answer.return_value = {"status": "answered", "claims": [], "missing_information": []}
        self.factory = Mock(side_effect=lambda *a: KeywordRetriever(load_catalog_chunks(self.catalog)))
        self.shopping = ShoppingAssistant(extractor=self.extractor, answerer=self.answerer,
            retriever_factory=self.factory, db_path=self.catalog, review_mode="rules")
        self.dialogue = Mock()
        self.dialogue.extract.side_effect = lambda *a, **kw: ({"needs": copy.deepcopy(self.needs),
            "options": {"price_ceiling": None, "excluded_product_ids": []}, "focus": None,
            "detail": False, "new_search": False}, {"cache_hit": True, "requests": 0})
        self.db = self.root / "graph.sqlite"

    def assistant(self):
        return GraphAssistant(self.db, self.shopping, self.dialogue)

    def run_graph(self, query="买跑步鞋", **kwargs):
        return self.assistant().run("session", query, method="keyword", **kwargs)

    def test_real_nodes_and_result_parity_with_original(self):
        old = self.shopping.run("买跑步鞋", method="keyword")
        events = []
        new = self.run_graph(on_event=events.append)
        for key in ("status", "requirements", "candidates", "answer", "steps"):
            self.assertEqual(old[key], new[key], key)
        self.assertEqual([e["node"] for e in events], ["requirements", "product_search", "evidence_retrieval", "recommendation", "commit_turn"])
        self.assertEqual(new["candidates"][0]["sku"]["sku_id"], "s_39")
        self.assertTrue(new["checkpoint_saved"])

    def test_clarification_pauses_then_new_instance_resumes(self):
        self.needs["clarification_questions"] = ["买什么类型？"]
        first = self.run_graph()
        self.assertTrue(first["paused"])
        self.assertEqual(first["pending_nodes"], ["human_input"])
        self.assertFalse(first["state_saved"])
        self.factory.assert_not_called()
        self.needs["clarification_questions"] = []
        second = self.run_graph("跑步鞋")
        self.assertEqual(second["status"], "completed")
        self.assertEqual(second["turn"], 2)
        self.assertEqual(self.extractor.extract.call_count, 1)
        self.assertEqual(self.dialogue.extract.call_count, 1)

    def test_empty_products_do_not_relax_budget(self):
        self.needs["max_price"] = 10
        first = self.run_graph()
        self.assertEqual(first["status"], "no_products")
        self.assertTrue(first["paused"])
        self.assertEqual(first["requirements"]["max_price"], 10)
        self.answerer.answer.assert_not_called()
        self.needs["max_price"] = 300
        second = self.run_graph("预算改为300元")
        self.assertEqual(second["status"], "completed")

    def test_failure_retry_does_not_repeat_completed_stages(self):
        self.answerer.answer.side_effect = [OnlineRequired("offline miss"), {"status": "answered", "claims": [], "missing_information": []}]
        first = self.run_graph()
        self.assertEqual(first["status"], "generation_disabled")
        self.assertEqual(first["pending_nodes"], ["recommendation"])
        self.assertTrue(first["resumable"])
        self.assertEqual(self.answerer.answer.call_count, 1)
        self.assertEqual(self.assistant().run("session", inspect_only=True)["status"], "failed")
        with self.assertRaises(ValueError):
            self.run_graph("新问题")
        result = self.assistant().run("session", retry=True, online=True)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(self.extractor.extract.call_count, 1)
        self.assertEqual(self.factory.call_count, 1)
        self.assertEqual(self.answerer.answer.call_args.kwargs["online"], True)

    def test_first_node_failure_can_resume(self):
        self.extractor.extract.side_effect = [OnlineRequired("no cache"), (self.needs, {"requests": 0})]
        first = self.run_graph()
        self.assertEqual(first["failed_stage"], "requirements")
        result = self.assistant().run("session", retry=True)
        self.assertEqual(result["status"], "completed")

    def test_sqlite_locked_is_retried_but_invalid_answer_is_not(self):
        from app.shopping import find_candidates
        products = find_candidates(self.needs, self.catalog)
        with patch("app.shopping.find_candidates", side_effect=[sqlite3.OperationalError("database is locked"), products]) as search:
            self.assertEqual(self.run_graph()["status"], "completed")
        self.assertEqual(search.call_count, 2)
        self.assertEqual(self.factory.call_count, 1)
        self.answerer.answer.return_value = {"status": "invalid_answer", "error": "bad citation"}
        first = self.run_graph("再看看")
        self.assertEqual(first["status"], "invalid_answer")
        self.assertEqual(self.answerer.answer.call_count, 2)

    def test_api_capable_node_is_not_automatically_retried_on_cache_write_error(self):
        self.extractor.extract.side_effect = sqlite3.OperationalError("database is locked")
        first = self.run_graph()
        self.assertEqual(first["status"], "failed")
        self.assertTrue(first["resumable"])
        self.assertEqual(self.extractor.extract.call_count, 1)

    def test_exhausted_local_retry_keeps_search_checkpoint(self):
        with patch("app.shopping.find_candidates", side_effect=sqlite3.OperationalError("database is locked")) as search:
            first = self.run_graph()
        self.assertEqual(search.call_count, 2)
        self.assertEqual(first["status"], "failed")
        self.assertEqual(first["pending_nodes"], ["product_search"])
        self.assertEqual(self.assistant().run("session", retry=True)["status"], "completed")
        self.assertEqual(self.extractor.extract.call_count, 1)

    def test_completed_conversation_restores_and_sessions_are_isolated(self):
        self.run_graph()
        second = self.run_graph("第一双怎么样")
        self.assertEqual(second["turn"], 2)
        self.assertEqual(self.dialogue.extract.call_args.args[1]["state"]["last_candidates"][0]["product_id"], "p_shoe")
        other = self.assistant().run("other", "买鞋", method="keyword")
        self.assertEqual(other["turn"], 1)
        self.assertEqual(self.extractor.extract.call_count, 2)

    def test_detail_followup_keeps_focus_and_answer_guard_scope(self):
        first = self.run_graph()
        target = first["candidates"][0]
        self.dialogue.extract.side_effect = None
        self.dialogue.extract.return_value = ({"needs": self.needs,
            "options": {"price_ceiling": None, "excluded_product_ids": []},
            "focus": target, "detail": True, "new_search": False}, {"requests": 0})
        second = self.run_graph("第一双的尺码呢")
        self.assertEqual(second["focus_product_id"], target["product_id"])
        scope = self.answerer.answer.call_args.kwargs["scope"]
        self.assertEqual(scope["shopping_focus_product"], target["product_id"])
        self.assertEqual(scope["shopping_allowed_products"], [target["product_id"]])

    def test_empty_result_keeps_last_displayed_cards_for_followup(self):
        first = self.run_graph()
        self.needs["max_price"] = 10
        self.assertTrue(self.run_graph("预算改成10元")["paused"])
        self.needs["max_price"] = 300
        self.run_graph("第一双呢")
        snapshot = self.dialogue.extract.call_args.args[1]
        self.assertEqual(snapshot["state"]["last_candidates"], first["candidates"])
        self.assertEqual(snapshot["state"]["requirements"]["max_price"], 10)

    def test_pending_review_cannot_be_bypassed_by_changing_mode(self):
        self.needs["preferences"] = ["日常通勤"]
        self.shopping.review_mode = "semantic"
        reviewer = Mock()
        reviewer.review.return_value = {"status": "invalid_review", "error": "bad quote"}
        self.shopping.reviewer = reviewer
        first = self.run_graph()
        self.assertEqual(first["failed_stage"], "fit_review")
        self.assertTrue(first["review_pending"])
        self.shopping.review_mode = "rules"
        again = self.assistant().run("session", retry=True)
        self.assertEqual(again["status"], "invalid_answer")
        self.assertEqual(reviewer.review.call_count, 2)
        self.answerer.answer.assert_not_called()
        self.assertEqual(self.factory.call_count, 1)

    def test_reference_without_candidates_pauses_without_model_call(self):
        self.assertTrue(self.run_graph("第二双怎么样")["paused"])
        self.extractor.extract.assert_not_called()

    def test_semantic_success_reuses_validated_matrix_without_answer_llm(self):
        from app.fit_review import guarded_checks
        self.needs["preferences"] = ["日常通勤"]
        self.shopping.review_mode = "semantic"
        reviewer = Mock()
        def review(needs, cards, hits, **kwargs):
            passage = next(hit for hit in hits if hit["section"] == "商品介绍")
            raw = {"checks": [{"product_id": "p_shoe", "requirement": "日常通勤", "status": "supported",
                "reason": "资料支持通勤用途。", "conditions": [], "citations": [{"chunk_id": passage["chunk_id"],
                    "quote": "适合日常通勤。", "applicability": "product"}]}]}
            checks = guarded_checks(raw, [{"product_id": "p_shoe", "requirement": "日常通勤"}], cards, hits)
            return {"status": "reviewed", "checks": checks, "cache_hit": True, "llm_stats": {"requests": 0}, "method": "llm_evidence_review_v1"}
        reviewer.review.side_effect = review
        self.shopping.reviewer = reviewer
        old = self.shopping.run("买跑步鞋", method="keyword")
        new = self.run_graph()
        self.assertEqual(new["answer"], old["answer"])
        self.assertEqual(new["candidates"], old["candidates"])
        self.answerer.answer.assert_not_called()
        self.assertFalse(new["review_pending"])

    def test_lock_prevents_concurrent_same_session(self):
        with thread_lock(self.db, "session"):
            with self.assertRaises((ValueError, BlockingIOError)):
                self.run_graph()

    def test_checkpoint_is_readable_from_another_process(self):
        self.needs["clarification_questions"] = ["买什么？"]
        self.run_graph()
        script = "from app.graph_workflow import GraphAssistant; import json,sys; print(json.dumps(GraphAssistant(sys.argv[1]).run('session', inspect_only=True)))"
        completed = subprocess.run([sys.executable, "-X", "utf8", "-B", "-c", script, str(self.db)],
            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, encoding="utf-8", timeout=30, check=True)
        result = json.loads(completed.stdout)
        self.assertTrue(result["paused"])
        self.assertEqual(result["pending_nodes"], ["human_input"])

    def test_input_validation_and_empty_state(self):
        self.assertEqual(self.assistant().run("session", inspect_only=True)["status"], "empty")
        for query in ("", " " , "a" * 2001):
            with self.assertRaises(ValueError):
                self.run_graph(query)
        with self.assertRaises(ValueError):
            self.assistant().run("../bad", "买鞋")
        with self.assertRaises(ValueError):
            self.assistant().run("session", retry=True)


if __name__ == "__main__":
    unittest.main()
