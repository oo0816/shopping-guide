"""Grounding and failure checks; all model responses are simulated."""

import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from app.llm import LLMConfig, LLMError, response_json
from app.rag import GroundedAnswerer, GroundingError, validate_answer


def evidence():
    return {
        "chunk_id": "a::memory", "mtm": "21AA000001", "model": "Sample",
        "text": "Sample / 21AA000001\n内存：出厂安装16GB内存，最高支持64GB。",
        "section": "内存", "source_url": "https://example.invalid/spec",
        "local_path": "data/raw/a.md", "line_start": 1, "line_end": 3,
        "market_status": "not_verified",
    }


def valid_answer():
    return {
        "status": "answered",
        "claims": [{"text": "21AA000001 出厂安装16GB内存。", "citations": [
            {"chunk_id": "a::memory", "quote": "出厂安装16GB内存"}]}],
        "missing_information": [],
    }


class GroundingTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.hit = evidence()
        self.retrieval = {"method": "test", "status": "matched", "hits": [self.hit]}
        self.client = Mock()
        self.client.config = LLMConfig("https://example.invalid/v1", "test-model", "fake-secret")
        self.client.stats = {"requests": 1}
        self.client.chat_json.return_value = valid_answer()
        self.answerer = GroundedAnswerer(self.client, Path(self.temp.name) / "answers.sqlite")

    def test_accepted_answer_has_reference_from_real_evidence(self):
        answer = self.answerer.answer("内存", self.retrieval, online=True)
        self.assertEqual("answered", answer["status"])
        self.assertEqual(self.hit["source_url"], answer["references"][0]["source_url"])
        self.assertFalse(answer["semantic_entailment_verified"])

    def test_invented_citation_or_quote_is_rejected(self):
        for field, value in (("chunk_id", "not-retrieved"), ("quote", "根本不存在的摘录")):
            data = valid_answer()
            data["claims"][0]["citations"][0][field] = value
            with self.subTest(field=field), self.assertRaises(GroundingError):
                validate_answer(data, [self.hit])

    def test_wrong_model_and_unsupported_capacity_are_rejected(self):
        for text in ("21BB000002 出厂16GB内存。", "21AA000001 出厂999GB内存。"):
            data = valid_answer()
            data["claims"][0]["text"] = text
            with self.subTest(text=text), self.assertRaises(GroundingError):
                validate_answer(data, [self.hit])

    def test_uncited_claim_and_inconsistent_refusal_are_rejected(self):
        data = valid_answer()
        data["claims"][0]["citations"] = []
        with self.assertRaises(GroundingError):
            validate_answer(data, [self.hit])
        data = valid_answer()
        data["status"] = "insufficient_evidence"
        with self.assertRaises(GroundingError):
            validate_answer(data, [self.hit])

    def test_empty_evidence_does_not_call_llm(self):
        answer = self.answerer.answer("价格", {"method": "test", "status": "no_match", "hits": []}, online=True)
        self.assertEqual("insufficient_evidence", answer["status"])
        self.client.chat_json.assert_not_called()

    def test_unknown_model_does_not_call_llm(self):
        answer = self.answerer.answer("21ZZ999999 内存",
            {"method": "test", "status": "unknown_model", "hits": []}, online=True)
        self.assertEqual("unknown_model", answer["status"])
        self.client.chat_json.assert_not_called()

    def test_nearest_neighbor_is_not_automatically_answerable(self):
        self.client.chat_json.return_value = {
            "status": "insufficient_evidence", "claims": [],
            "missing_information": ["没有售价资料。"]}
        answer = self.answerer.answer("售价", self.retrieval, online=True)
        self.assertEqual("insufficient_evidence", answer["status"])
        self.assertEqual([], answer["references"])

    def test_api_failure_remains_generation_failure(self):
        self.client.chat_json.side_effect = LLMError("LLM 网络请求失败。")
        answer = self.answerer.answer("内存", self.retrieval, online=True)
        self.assertEqual("generation_failed", answer["status"])
        self.assertEqual(["a::memory"], answer["retrieved_chunk_ids"])

    def test_invalid_answer_is_not_cached_or_shown_as_success(self):
        self.client.chat_json.return_value = {"status": "answered"}
        answer = self.answerer.answer("内存", self.retrieval, online=True)
        self.assertEqual("invalid_answer", answer["status"])
        second = self.answerer.answer("内存", self.retrieval)
        self.assertEqual("generation_disabled", second["status"])

    def test_validated_answer_reuses_cache_without_llm_call(self):
        self.answerer.answer("内存", self.retrieval, online=True)
        self.client.chat_json.reset_mock()
        answer = self.answerer.answer("内存", self.retrieval)
        self.assertTrue(answer["cache_hit"])
        self.assertEqual(0, answer["llm_stats"]["requests"])
        self.client.chat_json.assert_not_called()

    def test_changed_evidence_does_not_reuse_old_answer(self):
        self.answerer.answer("内存", self.retrieval, online=True)
        changed = copy.deepcopy(self.retrieval)
        changed["hits"][0]["text"] = "Sample / 21AA000001\n内存：出厂安装32GB内存。"
        answer = self.answerer.answer("内存", changed)
        self.assertEqual("generation_disabled", answer["status"])

    def test_offline_cache_miss_never_calls_llm(self):
        answer = self.answerer.answer("new question", self.retrieval)
        self.assertEqual("generation_disabled", answer["status"])
        self.client.chat_json.assert_not_called()

    def test_truncated_empty_and_non_json_model_output_are_rejected(self):
        for finish, text in (("length", '{"status":'), ("stop", ""), ("stop", "not json")):
            with self.subTest(finish=finish, text=text), self.assertRaises(LLMError):
                response_json({"choices": [{"finish_reason": finish, "message": {"content": text}}]})

    def test_secret_is_not_put_in_prompt_or_public_profile(self):
        self.answerer.answer("内存", self.retrieval, online=True)
        messages = self.client.chat_json.call_args.args[0]
        self.assertNotIn("fake-secret", json.dumps(messages))
        self.assertNotIn("fake-secret", repr(self.client.config))
        self.assertNotIn("fake-secret", json.dumps(self.client.config.profile))


if __name__ == "__main__":
    unittest.main()
