"""Select the original laptop corpus, imported products, or their union."""

import json
from app.catalog import DEFAULT_DB, load_catalog_chunks
from app.retrieval import DEFAULT_CORPUS, ROOT, load_chunks


def index_path(dataset):
    filename = "vector_index.json" if dataset == "original" else f"vector_index_{dataset}.json"
    return ROOT / "data/processed" / filename


def load_dataset(dataset="all"):
    if dataset not in ("original", "products", "all"):
        raise ValueError("未知资料集。")
    original, products = [], []
    original_scope = {}
    if dataset == "original" or (dataset == "all" and DEFAULT_CORPUS.is_file()):
        original = load_chunks()
        scope_path = ROOT / "data/catalog/dataset.json"
        original_scope = json.loads(scope_path.read_text(encoding="utf-8")) if scope_path.is_file() else {}
    if dataset == "products" or (dataset == "all" and DEFAULT_DB.is_file()):
        products = load_catalog_chunks()
    chunks = original + products
    if not chunks:
        raise ValueError("资料为空，请先运行 scripts/import_products.py 或 scripts/prepare_corpus.py。")
    if len({chunk["chunk_id"] for chunk in chunks}) != len(chunks):
        raise ValueError("合并后的片段 ID 重复。")
    if not products:
        return chunks, original_scope
    scope = {"trust_mode": "assumed_trusted", "prices_available": True,
             "product_library_available": True}
    if original:
        scope["original_laptop_scope"] = original_scope
    return chunks, scope
