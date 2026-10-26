"""Boundary checks that prevent cross-model evidence attribution."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from app.retrieval import KeywordRetriever, load_chunks


def chunk(chunk_id, mtm, text):
    return {
        "chunk_id": chunk_id, "mtm": mtm, "product_id": "p_" + mtm,
        "section": "规格", "text": f"笔记本，型号 {mtm}。\n{text}",
        "source_url": "https://example.invalid/spec",
    }


class RetrievalSafetyTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            chunk("a_memory", "21AA000001", "内存：出厂安装16GB内存。"),
            chunk("a_storage", "21AA000001", "存储：安装512GB SSD。"),
            chunk("b_memory", "21BB000002", "内存：出厂安装32GB内存。"),
            chunk("b_cpu", "21BB000002", "处理器：Intel Core 7。"),
        ]
        self.retriever = KeywordRetriever(self.rows)

    def test_unknown_model_does_not_borrow_other_evidence(self):
        result = self.retriever.search("21ZZ999999 内存是多少？")
        self.assertEqual("unknown_model", result["status"])
        self.assertEqual([], result["hits"])

    def test_model_match_is_case_insensitive_without_chinese_separator(self):
        result = self.retriever.search("21aa000001内存")
        self.assertEqual(["a_memory"], [hit["chunk_id"] for hit in result["hits"]])

    def test_fullwidth_identifier_normalizes(self):
        result = self.retriever.search("２１ＡＡ０００００１ 内存")
        self.assertEqual(["a_memory"], [hit["chunk_id"] for hit in result["hits"]])

    def test_explicit_multiple_models_are_retained(self):
        result = self.retriever.search("21AA000001 和21BB000002 内存")
        self.assertEqual({"a_memory", "b_memory"}, {hit["chunk_id"] for hit in result["hits"]})

    def test_model_header_alone_is_not_topical_evidence(self):
        result = self.retriever.search("21AA000001 充电续航")
        self.assertEqual([], result["hits"])

    def test_cpu_and_storage_vocabulary_are_supported(self):
        self.assertEqual("b_cpu", self.retriever.search("CPU", 1)["hits"][0]["chunk_id"])
        self.assertEqual("a_storage", self.retriever.search("硬盘", 1)["hits"][0]["chunk_id"])

    def test_invalid_query_and_limits_fail(self):
        for query, limit in (("", 3), ("内存", 0), ("内存", 21), ("内存", True)):
            with self.subTest(query=query, limit=limit), self.assertRaises(ValueError):
                self.retriever.search(query, limit)

    def test_duplicate_chunk_ids_are_rejected(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "chunks.jsonl"
            path.write_text("\n".join(json.dumps(self.rows[0]) for _ in range(2)), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_chunks(path)


if __name__ == "__main__":
    unittest.main()
