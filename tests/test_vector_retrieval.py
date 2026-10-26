"""Verify cache isolation, cost controls and model-specific evidence retrieval."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from app.embedding import EmbeddingClient, EmbeddingConfig, EmbeddingError, parse_vectors
from app.vector_retrieval import VectorRetriever, build_index
from test_retrieval import chunk


class VectorTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.config = EmbeddingConfig("https://example.invalid/v1", "test-model", 2, "fake-secret")
        self.client = EmbeddingClient(self.config, self.directory / "cache.sqlite")
        self.rows = [chunk("a", "21AA000001", "内存16GB"),
                     chunk("b", "21BB000002", "内存32GB")]
        self.index = self.directory / "index.json"

    def build(self):
        with patch.object(self.client, "_fetch", return_value=[[1.0, 0.0], [0.0, 1.0]]):
            build_index(self.rows, self.client, self.index, online=True)

    def test_default_cache_miss_never_uses_network(self):
        with patch.object(self.client, "_fetch") as fetch:
            with self.assertRaises(EmbeddingError):
                self.client.embed(["new query"])
            fetch.assert_not_called()

    def test_repeated_text_and_repeated_runs_reuse_cache(self):
        with patch.object(self.client, "_fetch", return_value=[[1.0, 0.0]]) as fetch:
            self.client.embed(["same", "same"], online=True)
            self.assertEqual(1, fetch.call_count)
            self.assertEqual(["same"], fetch.call_args.args[0])
        with patch.object(self.client, "_fetch") as fetch:
            self.assertEqual([[1.0, 0.0]], self.client.embed(["same"]))
            fetch.assert_not_called()

    def test_provider_profile_changes_do_not_reuse_old_vectors(self):
        with patch.object(self.client, "_fetch", return_value=[[1.0, 0.0]]):
            self.client.embed(["same"], online=True)
        other = EmbeddingClient(
            EmbeddingConfig("https://example.invalid/v1", "new-model", 2),
            self.directory / "cache.sqlite")
        with self.assertRaises(EmbeddingError):
            other.embed(["same"])

    def test_request_batches_are_at_most_ten_texts(self):
        def fetch(texts):
            return [[1.0, 0.0] for _ in texts]
        with patch.object(self.client, "_fetch", side_effect=fetch) as fetch:
            self.client.embed([f"text {i}" for i in range(11)], online=True)
        self.assertEqual([10, 1], [len(call.args[0]) for call in fetch.call_args_list])

    def test_response_order_is_restored_and_invalid_vectors_rejected(self):
        self.assertEqual([[1, 0], [0, 1]], parse_vectors(
            {"data": [{"index": 1, "embedding": [0, 1]}, {"index": 0, "embedding": [1, 0]}]}, 2, 2))
        for data in (
            {"data": [{"index": 0, "embedding": [0, 0]}]},
            {"data": [{"index": 0, "embedding": [float("nan"), 1]}]},
            {"data": [{"index": 0, "embedding": [1]}]},
            {"data": [{"index": 0, "embedding": [1, 0]}, {"index": 0, "embedding": [1, 0]}]},
        ):
            with self.subTest(data=data), self.assertRaises(EmbeddingError):
                parse_vectors(data, len(data["data"]), 2)

    def test_changed_corpus_requires_rebuild_without_network(self):
        self.build()
        changed = [{**self.rows[0], "text": "new content"}, self.rows[1]]
        with patch.object(self.client, "embed") as embed:
            with self.assertRaises(EmbeddingError):
                VectorRetriever(changed, self.client, self.index)
            embed.assert_not_called()

    def test_unknown_model_never_embeds_query_or_borrows_other_model(self):
        self.build()
        retriever = VectorRetriever(self.rows, self.client, self.index, online=True)
        with patch.object(self.client, "embed") as embed:
            self.assertEqual([], retriever.search("21ZZ999999 内存")["hits"])
            embed.assert_not_called()

    def test_model_filter_applies_before_result_ranking(self):
        self.build()
        retriever = VectorRetriever(self.rows, self.client, self.index)
        with patch.object(self.client, "embed", return_value=[[0.0, 1.0]]):
            result = retriever.search("21AA000001 内存")
        self.assertEqual(["a"], [hit["chunk_id"] for hit in result["hits"]])
        self.assertEqual("candidates", result["status"])

    def test_model_or_dimensions_change_requires_rebuild(self):
        self.build()
        other = EmbeddingClient(
            EmbeddingConfig("https://example.invalid/v1", "test-model", 3),
            self.directory / "cache.sqlite")
        with self.assertRaises(EmbeddingError):
            VectorRetriever(self.rows, other, self.index)

    def test_config_repr_and_profile_exclude_credentials(self):
        self.assertNotIn("fake-secret", repr(self.config))
        self.assertNotIn("fake-secret", json.dumps(self.config.profile))


if __name__ == "__main__":
    unittest.main()
