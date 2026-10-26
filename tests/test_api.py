"""HTTP entry tests with real GraphAssistant and isolated fixture databases."""
import json
import unittest
from unittest.mock import Mock

try:
    from fastapi.testclient import TestClient
    from app.api import create_app, public_result
    from app.graph_workflow import GraphAssistant, thread_lock
except ModuleNotFoundError as exc:
    if not exc.name.startswith(("fastapi", "langgraph")):
        raise
    TestClient = None


@unittest.skipIf(TestClient is None, "Install web/graph requirements in .venv")
class ApiTests(unittest.TestCase):
    def setUp(self):
        # Use the graph's real catalog fixture without running its tests twice.
        from test_graph_workflow import GraphTests
        GraphTests.setUp(self)
        self.client = TestClient(create_app(lambda: GraphAssistant(self.db, self.shopping, self.dialogue)))
        self.addCleanup(self.client.close)

    def chat(self, query="买跑步鞋", **kwargs):
        return self.client.post("/api/chat", json={"session_id": "web-test", "query": query, "method": "keyword", **kwargs})

    def test_page_assets_and_health_are_available_without_models(self):
        for path in ("/", "/assets/style.css", "/assets/app.js", "/api/health", "/openapi.json"):
            self.assertEqual(self.client.get(path).status_code, 200, path)
        self.assertIn("有据选购", self.client.get("/").text)
        self.assertIn("script-src 'self'", self.client.get("/").headers["content-security-policy"])
        self.extractor.extract.assert_not_called()

    def test_chat_returns_actual_candidates_and_opt_in_defaults_off(self):
        response = self.chat()
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["candidates"][0]["sku"]["sku_id"], "s_39")
        self.assertFalse(self.extractor.extract.call_args.kwargs["online"])
        self.assertNotIn("checkpoint_id", result)

    def test_history_is_loaded_from_a_new_graph_instance(self):
        self.chat()
        self.chat("第一双怎么样")
        result = self.client.get("/api/sessions/web-test").json()
        self.assertEqual(len(result["history"]), 2)
        self.assertEqual(result["history"][1]["query"], "第一双怎么样")
        self.assertEqual(result["turn"], 2)
        self.assertEqual(self.client.get("/api/sessions/other").json()["history"], [])
        self.assertEqual(self.extractor.extract.call_count, 1)

    def test_pause_then_supplement_through_http(self):
        self.needs["clarification_questions"] = ["买哪类？"]
        first = self.chat().json()
        self.assertTrue(first["paused"])
        self.assertEqual(self.client.get("/api/sessions/web-test").json()["history"], [])
        self.needs["clarification_questions"] = []
        second = self.chat("买跑步鞋").json()
        self.assertFalse(second["paused"])
        self.assertEqual(second["status"], "completed")
        history = self.client.get("/api/sessions/web-test").json()["history"]
        self.assertEqual([turn["result"]["status"] for turn in history], ["needs_clarification", "completed"])

    def test_failure_and_explicit_retry_preserve_completed_nodes(self):
        from app.shopping import OnlineRequired
        self.answerer.answer.side_effect = [OnlineRequired("no cache"), {"status": "answered", "claims": [], "missing_information": []}]
        self.assertTrue(self.chat().json()["resumable"])
        self.assertEqual(self.chat("不能覆盖").status_code, 409)
        retried = self.client.post("/api/retry", json={"session_id": "web-test", "online": True})
        self.assertEqual(retried.status_code, 200)
        self.assertEqual(retried.json()["status"], "completed")
        self.assertEqual(self.factory.call_count, 1)
        self.assertEqual(self.extractor.extract.call_count, 1)
        self.assertTrue(self.answerer.answer.call_args.kwargs["online"])

    def test_request_validation_rejects_bad_input_before_models(self):
        for payload in (
            {"query": ""}, {"query": " "}, {"query": "x" * 2001}, {"session_id": "../bad"},
            {"online": "true"}, {"online": 1}, {"method": "fake"}, {"api_key": "do-not-accept"},
        ):
            self.assertEqual(self.chat(**payload).status_code, 422, payload)
        self.assertEqual(self.client.get("/api/sessions/" + "a" * 65).status_code, 422)
        self.extractor.extract.assert_not_called()

    def test_retry_does_not_accept_new_question(self):
        self.assertEqual(self.client.post("/api/retry", json={"session_id": "web-test", "query": "new"}).status_code, 422)
        self.assertEqual(self.client.post("/api/retry", json={"session_id": "web-test"}).status_code, 409)

    def test_same_session_concurrent_request_returns_conflict(self):
        with thread_lock(self.db, "web-test"):
            self.assertEqual(self.chat().status_code, 409)
        self.extractor.extract.assert_not_called()

    def test_cross_origin_and_untrusted_host_cannot_trigger_models(self):
        self.assertEqual(self.client.post("/api/chat", headers={"Origin": "https://example.invalid"}, json={"session_id": "x", "query": "买鞋"}).status_code, 403)
        self.assertEqual(self.client.get("/api/health", headers={"Host": "example.invalid"}).status_code, 400)
        self.assertEqual(self.client.get("/api/health", headers={"Origin": "http://testserver"}).status_code, 200)
        self.extractor.extract.assert_not_called()

    def test_history_is_not_cached_by_browser(self):
        response = self.client.get("/api/sessions/web-test")
        self.assertEqual(response.headers["cache-control"], "no-store")

    def test_storage_errors_do_not_return_local_paths(self):
        factory = Mock()
        factory.return_value.run.side_effect = OSError("C:/private/path")
        with TestClient(create_app(factory)) as client:
            result = client.get("/api/sessions/x")
        self.assertEqual(result.status_code, 503)
        self.assertNotIn("private", result.text)

    def test_internal_model_and_source_fields_are_not_exposed(self):
        result = public_result({"status": "partial", "checkpoint_id": "private", "fit_review": {"model_checks": ["private"]},
            "candidates": [{"product_id": "p_one", "title": "<script>alert(1)</script>", "local_path": "private", "retrieval_score": 1}],
            "answer": {"claims": [{"text": "<script>", "citations": [{"chunk_id": "c", "quote": "<script>"}]}], "references": [{"local_path": "private"}]}})
        self.assertNotIn("private", json.dumps(result))
        self.assertEqual(result["candidates"][0]["title"], "<script>alert(1)</script>")


if __name__ == "__main__":
    unittest.main()
