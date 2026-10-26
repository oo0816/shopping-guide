"""Import product JSON into a separate SQLite catalog and searchable passages."""

from contextlib import closing
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re
import sqlite3

from app.retrieval import ROOT, normalize

DEFAULT_SOURCE = ROOT / "data/imports/product-library"
DEFAULT_DB = ROOT / "data/processed/product_catalog.sqlite"
PRODUCT_ID = re.compile(r"\bp_[a-z0-9_]+\b", re.I)
SCHEMA = """
CREATE TABLE IF NOT EXISTS products (
    product_id TEXT PRIMARY KEY, title TEXT NOT NULL, brand TEXT NOT NULL,
    category TEXT NOT NULL, sub_category TEXT NOT NULL,
    base_price_cents INTEGER NOT NULL, local_path TEXT NOT NULL,
    content_sha256 TEXT NOT NULL, search_text TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS skus (
    sku_id TEXT PRIMARY KEY, product_id TEXT NOT NULL REFERENCES products(product_id),
    properties_json TEXT NOT NULL, price_cents INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS skus_by_product ON skus(product_id, price_cents);
CREATE TABLE IF NOT EXISTS knowledge_chunks (
    chunk_id TEXT PRIMARY KEY, product_id TEXT NOT NULL REFERENCES products(product_id),
    payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS catalog_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


def money_cents(value):
    if isinstance(value, bool) or value is None:
        raise ValueError("价格必须是非负金额，最多两位小数。")
    try:
        amount = Decimal(str(value))
        if not amount.is_finite() or amount < 0 or amount > Decimal("1000000000"):
            raise ValueError("价格超出允许范围。")
        cents = amount * 100
        if cents != cents.to_integral_value():
            raise ValueError("价格最多保留两位小数。")
        return int(cents)
    except (InvalidOperation, TypeError):
        raise ValueError("价格格式无效。") from None


def required_text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(field + " 必须是非空文本。")
    return value.strip()


def readable_path(path):
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def make_chunk(product, suffix, section, pointer, body, local_path):
    product_id = product["product_id"]
    identity = f"商品：{product['title']}。品牌：{product['brand']}。品类：{product['category']} / {product['sub_category']}。"
    return {
        "chunk_id": f"catalog::{product_id}::{suffix}", "product_id": product_id,
        "mtm": "", "model": product["title"], "section": section,
        "text": f"商品编号：{product_id}\n{identity}\n{body}",
        "source_url": "", "local_path": local_path, "json_pointer": pointer,
        "market_status": "catalog", "evidence_kind": "product_library",
        "assumed_trusted": True,
    }


def prepare_product(path):
    raw = path.read_bytes()
    product = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(product, dict):
        raise ValueError("商品 JSON 必须是对象。")
    for field in ("product_id", "title", "brand", "category", "sub_category"):
        product[field] = required_text(product.get(field), field)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", product["product_id"]):
        raise ValueError("商品 ID 含无效字符。")
    base_price = money_cents(product.get("base_price"))
    skus = product.get("skus")
    knowledge = product.get("rag_knowledge")
    if not isinstance(skus, list) or not skus or not isinstance(knowledge, dict):
        raise ValueError("商品必须包含 SKU 列表和 rag_knowledge 对象。")
    local_path = readable_path(path)
    chunks = [make_chunk(product, "overview", "商品信息", "/",
                         f"商品基础价：{base_price / 100:.2f}元。具体规格价格见对应 SKU。", local_path)]
    normalized_skus = []
    for index, sku in enumerate(skus):
        if not isinstance(sku, dict):
            raise ValueError("SKU 必须是对象。")
        sku_id = required_text(sku.get("sku_id"), "sku_id")
        properties = sku.get("properties")
        if not isinstance(properties, dict) or not properties or any(
            not isinstance(key, str) or not isinstance(value, str) for key, value in properties.items()
        ):
            raise ValueError("SKU properties 必须是非空的文本键值对象。")
        price = money_cents(sku.get("price"))
        normalized_skus.append((sku_id, product["product_id"], json.dumps(properties, ensure_ascii=False), price))
        body = "规格：" + "；".join(f"{key}：{value}" for key, value in properties.items())
        body += f"。该规格价格：{price / 100:.2f}元。"
        chunks.append(make_chunk(product, f"sku_{index + 1}", "SKU 规格与价格",
                                 f"/skus/{index}", body, local_path))
    description = knowledge.get("marketing_description", "")
    if description:
        chunks.append(make_chunk(product, "description", "商品介绍",
                                 "/rag_knowledge/marketing_description",
                                 required_text(description, "marketing_description"), local_path))
    for field in ("official_faq", "user_reviews"):
        values = knowledge.get(field, [])
        if not isinstance(values, list):
            raise ValueError(field + " 必须是列表。")
        for index, value in enumerate(values):
            if not isinstance(value, dict):
                raise ValueError(field + " 条目必须是对象。")
            if field == "official_faq":
                body = "问题：" + required_text(value.get("question"), "question")
                body += "\n回答：" + required_text(value.get("answer"), "answer")
                section, suffix = "商品问答", f"faq_{index + 1}"
            else:
                rating = value.get("rating")
                if type(rating) is not int or not 1 <= rating <= 5:
                    raise ValueError("评价分数必须为 1～5 的整数。")
                body = f"评分：{rating}/5。\n评价：" + required_text(value.get("content"), "content")
                section, suffix = "商品评价", f"review_{index + 1}"
            chunks.append(make_chunk(product, suffix, section,
                                     f"/rag_knowledge/{field}/{index}", body, local_path))
    search_text = normalize(" ".join(
        [product[key] for key in ("title", "brand", "category", "sub_category")] +
        [str(value) for sku in skus for value in sku["properties"].values()]
    ))
    row = tuple(product[key] for key in ("product_id", "title", "brand", "category", "sub_category"))
    row += (base_price, local_path, hashlib.sha256(raw).hexdigest(), search_text)
    return row, normalized_skus, chunks


def import_catalog(source=DEFAULT_SOURCE, db_path=DEFAULT_DB):
    source, db_path = Path(source).resolve(), Path(db_path)
    paths = sorted(source.glob("*/data/*.json"))
    if not paths:
        raise ValueError("未找到商品 JSON，请检查 data/imports/product-library。")
    prepared = []
    for path in paths:
        resolved = path.resolve()
        if not resolved.is_relative_to(source):
            raise ValueError("商品文件超出指定目录。")
        try:
            prepared.append(prepare_product(resolved))
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"{path.name}：{exc}") from None
    products = [item[0] for item in prepared]
    skus = [sku for item in prepared for sku in item[1]]
    chunks = [chunk for item in prepared for chunk in item[2]]
    if len({row[0] for row in products}) != len(products):
        raise ValueError("商品 ID 重复，导入未执行。")
    if len({row[0] for row in skus}) != len(skus):
        raise ValueError("SKU ID 重复，导入未执行。")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(db_path, timeout=10)) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(SCHEMA)
        with connection:
            previous = dict(connection.execute("SELECT product_id, content_sha256 FROM products"))
            incoming = {row[0]: row[7] for row in products}
            connection.execute("DELETE FROM knowledge_chunks")
            connection.execute("DELETE FROM skus")
            connection.execute("DELETE FROM products")
            connection.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", products)
            connection.executemany("INSERT INTO skus VALUES (?, ?, ?, ?)", skus)
            connection.executemany("INSERT INTO knowledge_chunks VALUES (?, ?, ?)", [
                (chunk["chunk_id"], chunk["product_id"], json.dumps(chunk, ensure_ascii=False))
                for chunk in chunks
            ])
            connection.execute("INSERT OR REPLACE INTO catalog_settings VALUES ('trust_mode', 'assumed_trusted')")
            connection.execute("INSERT OR REPLACE INTO catalog_settings VALUES ('schema_version', '1')")
    return {
        "products": len(products), "skus": len(skus), "chunks": len(chunks),
        "created": len(incoming.keys() - previous.keys()),
        "updated": sum(key in previous and previous[key] != digest for key, digest in incoming.items()),
        "removed": len(previous.keys() - incoming.keys()),
        "trust_mode": "assumed_trusted",
    }


def connect_catalog(db_path=DEFAULT_DB):
    path = Path(db_path).resolve()
    if not path.is_file():
        raise ValueError("商品尚未导入，请先运行 scripts/import_products.py。")
    connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=10)
    connection.row_factory = sqlite3.Row
    return connection


def load_catalog_chunks(db_path=DEFAULT_DB):
    with closing(connect_catalog(db_path)) as connection:
        return [json.loads(row[0]) for row in connection.execute(
            "SELECT payload_json FROM knowledge_chunks ORDER BY chunk_id")]


def search_products(query="", *, category=None, brand=None, min_price=None,
                    max_price=None, product_id=None, sub_category=None, limit=10, db_path=DEFAULT_DB):
    if not isinstance(query, str) or type(limit) is not int or not 1 <= limit <= 50:
        raise ValueError("商品查询必须是文本，limit 必须为 1～50 的整数。")
    lower = money_cents(min_price) if min_price is not None else None
    upper = money_cents(max_price) if max_price is not None else None
    if lower is not None and upper is not None and lower > upper:
        raise ValueError("最低价不能高于最高价。")
    clauses, parameters = [], []
    for column, value in (("category", category), ("brand", brand), ("product_id", product_id), ("sub_category", sub_category)):
        if value is not None:
            clauses.append(f"p.{column} = ? COLLATE NOCASE")
            parameters.append(required_text(value, column))
    for term in normalize(query).split():
        clauses.append("p.search_text LIKE ? ESCAPE '\\'")
        parameters.append("%" + term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%")
    price_clauses, price_parameters = [], []
    if lower is not None:
        price_clauses.append("s.price_cents >= ?")
        price_parameters.append(lower)
    if upper is not None:
        price_clauses.append("s.price_cents <= ?")
        price_parameters.append(upper)
    sku_where = " AND ".join(["s.product_id = p.product_id"] + price_clauses)
    clauses.append("EXISTS (SELECT 1 FROM skus s WHERE " + sku_where + ")")
    with closing(connect_catalog(db_path)) as connection:
        rows = connection.execute(
            "SELECT p.* FROM products p WHERE " + " AND ".join(clauses),
            parameters + price_parameters).fetchall()
        items = []
        for row in rows:
            matched_skus = []
            for sku in connection.execute("SELECT * FROM skus WHERE product_id = ? ORDER BY price_cents, sku_id",
                                          (row["product_id"],)):
                if lower is not None and sku["price_cents"] < lower:
                    continue
                if upper is not None and sku["price_cents"] > upper:
                    continue
                matched_skus.append({"sku_id": sku["sku_id"],
                                     "properties": json.loads(sku["properties_json"]),
                                     "price": sku["price_cents"] / 100})
            items.append({
                "product_id": row["product_id"], "title": row["title"], "brand": row["brand"],
                "category": row["category"], "sub_category": row["sub_category"],
                "base_price": row["base_price_cents"] / 100,
                "matched_skus": matched_skus, "local_path": row["local_path"],
                "trust_mode": "assumed_trusted",
            })
    items.sort(key=lambda item: (item["matched_skus"][0]["price"], item["product_id"]))
    return {"query": query, "total": len(items), "items": items[:limit],
            "filters": {"category": category, "brand": brand, "min_price": min_price,
                        "max_price": max_price, "product_id": product_id, "sub_category": sub_category},
            "trust_mode": "assumed_trusted"}


def catalog_info(db_path=DEFAULT_DB):
    with closing(connect_catalog(db_path)) as connection:
        return {
            "products": connection.execute("SELECT COUNT(*) FROM products").fetchone()[0],
            "skus": connection.execute("SELECT COUNT(*) FROM skus").fetchone()[0],
            "chunks": connection.execute("SELECT COUNT(*) FROM knowledge_chunks").fetchone()[0],
            "categories": dict(connection.execute("SELECT category, COUNT(*) FROM products GROUP BY category")),
            "trust_mode": "assumed_trusted",
        }
