"""Product import, SKU budgets and retrieval-to-answer integration, entirely offline."""

import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

from app.catalog import catalog_info, import_catalog, load_catalog_chunks, money_cents, search_products
from app.embedding import EmbeddingConfig
from app.llm import LLMConfig
from app.rag import GroundedAnswerer, GroundingError, materialize, validate_answer
from app.retrieval import KeywordRetriever, load_chunks
from app.vector_retrieval import VectorRetriever, build_index


def product(product_id="p_one"):
    return {
        "product_id": product_id, "title": "轻便运动鞋", "brand": "Sample",
        "category": "服饰", "sub_category": "运动鞋", "base_price": 10,
        "skus": [
            {"sku_id": product_id + "_small", "properties": {"尺码": "小码"}, "price": 20},
            {"sku_id": product_id + "_large", "properties": {"尺码": "大码"}, "price": 50},
        ],
        "rag_knowledge": {
            "marketing_description": "轻便，适合通勤。",
            "official_faq": [{"question": "能机洗吗？", "answer": "建议手洗，不建议机洗。"}],
            "user_reviews": [{"nickname": "A", "rating": 4, "content": "通勤穿着轻便。"}],
        },
    }


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "input"
        self.db = self.root / "catalog.sqlite"
        self.path = self.write(product())

    def write(self, value):
        folder = self.source / value["category"] / "data"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / (value["product_id"] + ".json")
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def run_import(self):
        return import_catalog(self.source, self.db)

    def search(self, query="", **kwargs):
        return search_products(query, db_path=self.db, **kwargs)

    def test_import_is_repeatable_and_preserves_raw_files(self):
        raw = self.path.read_bytes()
        first, second = self.run_import(), self.run_import()
        self.assertEqual((1, 2, 6), (first["products"], first["skus"], first["chunks"]))
        self.assertEqual(0, second["created"])
        self.assertEqual(0, second["updated"])
        self.assertEqual(raw, self.path.read_bytes())
        self.assertEqual("assumed_trusted", catalog_info(self.db)["trust_mode"])
        self.assertEqual(6, len(load_catalog_chunks(self.db)))

    def test_budget_checks_actual_skus_instead_of_base_price(self):
        self.run_import()
        self.assertEqual([], self.search(max_price="15")["items"])
        result = self.search("运动鞋", min_price="19.99", max_price="20.00")
        self.assertEqual(["p_one_small"], [s["sku_id"] for s in result["items"][0]["matched_skus"]])
        self.assertEqual([], self.search(min_price="20.01", max_price="49.99")["items"])

    def test_currency_validation_and_invalid_price_ranges(self):
        self.assertEqual(10, money_cents("0.10"))
        for value in (True, None, "-1", "NaN", "Infinity", "0.001"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                money_cents(value)
        self.run_import()
        with self.assertRaises(ValueError):
            self.search(min_price=30, max_price=20)

    def test_category_brand_and_literal_queries_cannot_escape_filters(self):
        second = product("p_two")
        second.update(title="轻便电脑", category="电脑", sub_category="笔记本", brand="Other")
        self.write(second)
        self.run_import()
        self.assertEqual(["p_one"], [p["product_id"] for p in self.search(category="服饰", brand="sample")["items"]])
        for query in ("%", "_", "' OR 1=1 --"):
            self.assertEqual([], self.search(query)["items"])
        self.assertEqual([], self.search(category="不存在")["items"])

    def test_invalid_import_does_not_replace_previous_catalog(self):
        self.run_import()
        broken = product()
        broken["skus"][0]["price"] = -1
        self.write(broken)
        with self.assertRaises(ValueError):
            self.run_import()
        self.assertEqual(20, self.search()["items"][0]["matched_skus"][0]["price"])

    def test_duplicate_product_and_sku_ids_fail_before_writing(self):
        self.run_import()
        second = product("p_two")
        second["skus"][0]["sku_id"] = "p_one_small"
        second_path = self.write(second)
        with self.assertRaises(ValueError):
            self.run_import()
        self.assertEqual(1, catalog_info(self.db)["products"])
        second_path.unlink()
        other_folder = self.source / "重复" / "data"
        other_folder.mkdir(parents=True)
        (other_folder / "copy.json").write_bytes(self.path.read_bytes())
        with self.assertRaises(ValueError):
            self.run_import()
        self.assertEqual(1, catalog_info(self.db)["products"])

    def test_reimport_updates_changed_values_and_removes_deleted_products(self):
        second_path = self.write(product("p_two"))
        self.run_import()
        changed = product()
        changed["skus"][0]["price"] = 25
        self.write(changed)
        second_path.unlink()
        result = self.run_import()
        self.assertEqual((1, 1), (result["updated"], result["removed"]))
        self.assertEqual(1, catalog_info(self.db)["products"])
        self.assertEqual(25, self.search()["items"][0]["matched_skus"][0]["price"])

    def test_missing_database_query_does_not_create_empty_database(self):
        with self.assertRaises(ValueError):
            self.search()
        self.assertFalse(self.db.exists())

    def test_product_chunks_load_without_fabricated_mtm_or_source_url(self):
        self.run_import()
        chunks = load_catalog_chunks(self.db)
        corpus = self.root / "chunks.jsonl"
        corpus.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in chunks), encoding="utf-8")
        loaded = load_chunks(corpus)
        self.assertTrue(all(c["mtm"] == "" and c["source_url"] == "" for c in loaded))
        for chunk in loaded:
            self.assertEqual("p_one", chunk["product_id"])
            pointer = chunk["json_pointer"]
            value = json.loads(self.path.read_text(encoding="utf-8"))
            for part in pointer.split("/")[1:] if pointer != "/" else []:
                value = value[int(part)] if isinstance(value, list) else value[part]
            self.assertIsNotNone(value)

    def test_product_filter_prevents_cross_product_evidence_and_unknown_borrowing(self):
        self.write(product("p_two"))
        self.run_import()
        retriever = KeywordRetriever(load_catalog_chunks(self.db))
        result = retriever.search("机洗", product_ids={"p_one"})
        self.assertTrue(result["hits"])
        self.assertTrue(all(h["product_id"] == "p_one" for h in result["hits"]))
        unknown = retriever.search("机洗", product_ids={"p_missing"})
        self.assertEqual("unknown_product", unknown["status"])
        self.assertEqual([], unknown["hits"])

    def test_imported_passage_to_answer_and_offline_cache(self):
        self.run_import()
        chunks = load_catalog_chunks(self.db)
        result = KeywordRetriever(chunks).search("机洗", 1, product_ids={"p_one"})
        hit = result["hits"][0]
        self.assertEqual("商品问答", hit["section"])
        client = Mock()
        client.config = LLMConfig("https://example.invalid/v1", "test", "test-key")
        client.stats = {"requests": 1}
        client.chat_json.return_value = {
            "status": "answered", "claims": [{
                "text": "p_one 建议手洗，不建议机洗。",
                "citations": [{"chunk_id": hit["chunk_id"], "quote": "建议手洗，不建议机洗。"}],
            }], "missing_information": [],
        }
        answerer = GroundedAnswerer(client, self.root / "answers.sqlite")
        first = answerer.answer("能机洗吗", result, online=True, scope={"trust_mode": "assumed_trusted"})
        self.assertEqual("answered", first["status"])
        self.assertEqual("/rag_knowledge/official_faq/0", first["references"][0]["json_pointer"])
        self.assertEqual(str(self.path.resolve()), first["references"][0]["local_path"])
        self.assertIn("商品选购与资料问答助手", client.chat_json.call_args.args[0][0]["content"])
        client.chat_json.reset_mock()
        second = answerer.answer("能机洗吗", result, scope={"trust_mode": "assumed_trusted"})
        self.assertTrue(second["cache_hit"])
        client.chat_json.assert_not_called()
        wrong = copy.deepcopy(client.chat_json.return_value)
        wrong["claims"][0]["text"] = "p_other 建议手洗。"
        with self.assertRaises(GroundingError):
            validate_answer(wrong, result["hits"])

    def test_unknown_product_never_calls_llm(self):
        self.run_import()
        client = Mock()
        result = KeywordRetriever(load_catalog_chunks(self.db)).search("价格", product_ids={"p_missing"})
        answer = GroundedAnswerer(client).answer("价格", result, online=True)
        self.assertEqual("unknown_product", answer["status"])
        client.chat_json.assert_not_called()

    def test_product_vector_index_uses_same_chunks_and_filters_before_embedding(self):
        self.write(product("p_two"))
        self.run_import()
        chunks = load_catalog_chunks(self.db)
        client = Mock()
        client.config = EmbeddingConfig("https://example.invalid/v1", "fake", 2)
        client.stats = {"requests": 0}
        client.embed.side_effect = lambda texts, **kwargs: [[1.0, 0.0] for _ in texts]
        index = self.root / "vectors.json"
        build_index(chunks, client, path=index)
        retriever = VectorRetriever(chunks, client, index_path=index)
        result = retriever.search("机洗", product_ids={"p_two"})
        self.assertTrue(all(h["product_id"] == "p_two" for h in result["hits"]))
        client.embed.reset_mock()
        self.assertEqual("unknown_product", retriever.search("价格", product_ids={"p_missing"})["status"])
        client.embed.assert_not_called()


if __name__ == "__main__":
    unittest.main()
