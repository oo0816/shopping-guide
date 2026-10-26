"""Prepare traceable, section-based chunks from the small verified catalog."""

import csv
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SECTION = re.compile(r"^## (.+) \[([a-z_]+)\]$")


def build_chunks(root):
    with (root / "data/catalog/products.csv").open(encoding="utf-8-sig", newline="") as stream:
        products = list(csv.DictReader(stream))
    with (root / "data/catalog/sources.csv").open(encoding="utf-8-sig", newline="") as stream:
        sources = list(csv.DictReader(stream))
    product_map = {p["product_id"]: p for p in products}
    if len(product_map) != len(products) or not products:
        raise ValueError("商品 ID 重复或商品表为空。")
    if len({s["source_id"] for s in sources}) != len(sources):
        raise ValueError("来源 ID 重复。")
    chunks = []
    for source in sources:
        product_id = source["product_id"]
        if product_id not in product_map:
            raise ValueError("来源关联了不存在的商品。")
        path = (root / source["local_path"]).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError("来源文件不存在或超出项目目录。")
        product = product_map[product_id]
        lines = path.read_text(encoding="utf-8-sig").splitlines()
        if not any(line == f"型号编号：{product['mtm']}" for line in lines):
            raise ValueError("来源摘录中的型号编号与商品记录不一致。")
        headings = [(i, SECTION.fullmatch(line)) for i, line in enumerate(lines) if SECTION.fullmatch(line)]
        if not headings:
            raise ValueError("来源摘录缺少可识别的事实章节。")
        seen_sections = set()
        for pos, (line_index, match) in enumerate(headings):
            heading, slug = match.groups()
            if slug in seen_sections:
                raise ValueError("同一来源内章节 ID 重复。")
            seen_sections.add(slug)
            end = headings[pos + 1][0] if pos + 1 < len(headings) else len(lines)
            body = "\n".join(lines[line_index + 1:end]).strip()
            if not body:
                raise ValueError("事实章节为空。")
            chunks.append({
                "chunk_id": source["source_id"] + "::" + slug,
                "source_id": source["source_id"], "product_id": product_id,
                "mtm": product["mtm"], "model": product["model"],
                "section": heading,
                "text": f"{product['model']}；型号编号 {product['mtm']}。\n{heading}：{body}",
                "source_url": source["url"], "local_path": source["local_path"],
                "line_start": line_index + 1, "line_end": end,
                "collected_at": source["collected_at"],
                "evidence_kind": "manually_curated_official_search_excerpt",
                "market_status": product["market_status"],
            })
    if len({c["chunk_id"] for c in chunks}) != len(chunks):
        raise ValueError("片段 ID 重复。")
    return chunks


def main():
    try:
        chunks = build_chunks(ROOT)
        output = ROOT / "data/processed/chunks.jsonl"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in chunks), encoding="utf-8")
        print(f"资料检查通过，生成 {len(chunks)} 个片段：data/processed/chunks.jsonl")
        print("每个片段关联具体型号、来源和本地行号；未调用模型或 Embedding。")
    except (OSError, ValueError, KeyError, csv.Error):
        print("资料准备失败，请检查商品与来源关联、型号编号和章节格式。")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
