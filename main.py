"""Search products, retrieve passages, or generate answers with explicit online access."""

import argparse
import json
import sqlite3
import sys

from app.catalog import search_products
from app.shopping import ShoppingAssistant
from app.conversation import ConversationAssistant
from app.sessions import SessionStore
from app.corpus import index_path, load_dataset
from app.embedding import EmbeddingClient, load_config
from app.rag import GroundedAnswerer
from app.retrieval import KeywordRetriever
from app.vector_retrieval import VectorRetriever


def show_location(item):
    if item.get("source_url"):
        print("资料：" + item["source_url"])
    if item.get("json_pointer"):
        print(f"位置：{item['local_path']}，JSON {item['json_pointer']}")
    elif item.get("line_start") is not None:
        print(f"位置：{item['local_path']}，第 {item['line_start']}～{item['line_end']} 行")


def show_evidence(result):
    print(f"找到 {len(result['hits'])} 个候选片段。")
    for index, hit in enumerate(result["hits"], 1):
        identity = hit.get("mtm") or hit.get("product_id", "")
        print(f"\n[{index}] {hit['model']} / {identity} / {hit['section']}")
        print(hit["text"])
        print("片段：" + hit["chunk_id"])
        show_location(hit)
        if hit.get("market_status") == "withdrawn":
            print("状态：历史型号，厂商标注 Withdrawn。")


def show_answer(answer):
    labels = {"answered": "根据当前资料", "partial": "部分问题有资料支持",
              "insufficient_evidence": "当前证据不足", "unknown_model": "型号未收录",
              "unknown_product": "商品未收录", "needs_topic": "请补充问题"}
    print(labels.get(answer["status"], "回答生成未完成") + "：")
    references = answer.get("references", [])
    numbers = {reference["chunk_id"]: index for index, reference in enumerate(references, 1)}
    for claim in answer.get("claims", []):
        marks = "".join(f"[{index}]" for index in dict.fromkeys(
            numbers[citation["chunk_id"]] for citation in claim["citations"]))
        print("- " + claim["text"] + " " + marks)
    for item in answer.get("missing_information", []):
        print("- 尚缺信息：" + item)
    for index, reference in enumerate(references, 1):
        identity = reference.get("mtm") or reference.get("product_id", "")
        print(f"\n[{index}] {reference['model']} / {identity}")
        show_location(reference)
        if reference.get("market_status") == "withdrawn":
            print("历史型号：不能据此推断当前在售。")
    if answer.get("error"):
        print(answer["error"])
    print("\n回答缓存命中：" + str(answer.get("cache_hit", False)))
    print("LLM 统计：" + json.dumps(answer.get("llm_stats", {}), ensure_ascii=False))


def show_products(result):
    print(f"符合条件的商品：{result['total']} 个，展示 {len(result['items'])} 个。")
    for item in result["items"]:
        print(f"\n{item['title']} / {item['product_id']}")
        print(f"品牌：{item['brand']}；品类：{item['category']} / {item['sub_category']}")
        for sku in item["matched_skus"]:
            properties = "；".join(f"{key}：{value}" for key, value in sku["properties"].items())
            print(f"- {sku['sku_id']}：{properties}；{sku['price']:.2f}元")


def show_shopping(result):
    labels = {"completed": "选购结果", "partial": "部分偏好有资料支持",
              "needs_clarification": "还需要确认", "no_products": "没有符合条件的商品",
              "insufficient_evidence": "候选符合筛选条件，但推荐依据不足"}
    if result.get("session_id"):
        print(f"会话：{result['session_id']}；第 {result['turn']} 轮；已保存：{result['state_saved']}")
    print(labels.get(result["status"], "选购流程未完成") + "：")
    needs = result.get("requirements")
    if needs:
        print("商品类型：" + "、".join(needs["sub_categories"]) + (" " + needs["keywords"] if needs["keywords"] else ""))
        if needs["max_price"] is not None:
            print(f"预算上限：{needs['max_price']:.2f}元")
        if needs["preferences"]:
            print("偏好：" + "、".join(needs["preferences"]))
    if result.get("relative_price_ceiling") is not None:
        print(f"本轮继续限定价格不超过：{result['relative_price_ceiling']:.2f}元")
    for question in result.get("questions", []):
        print("- " + question)
    for condition in result.get("unsupported_constraints", []):
        print("- 暂时无法验证的硬条件：" + condition)
    if result.get("message"):
        print(result["message"])
    if result.get("review_pending"):
        print("适配审核未完成，以下仅为通过硬条件的候选。")
    for index, card in enumerate(result["candidates"], 1):
        sku = card["sku"]
        print(f"\n候选 {index}：{card['title']} / {card['product_id']}")
        print("规格：" + "；".join(f"{key}：{value}" for key, value in sku["properties"].items()))
        print(f"价格：{sku['price']:.2f}元；SKU：{sku['sku_id']}")
        if card.get("suitability"):
            print(card["suitability"]["label"])
            for signal in card["suitability"]["signals"]:
                if signal["effect"] == "caution":
                    print("资料限制：" + signal["quote"])
        for check in card.get("fit_review", []):
            label = {"supported": "支持", "limited": "有限制", "unknown": "未知"}[check["status"]]
            print(f"- {check['requirement']}：{label}；{check['reason']}")
            for condition in check["conditions"]:
                print("  适用条件或限制：" + condition)
            for note in check["guard_notes"]:
                print("  检查说明：" + note)
    if result.get("answer"):
        print()
        show_answer(result["answer"])
    if result.get("error"):
        print(result["error"])


def main():
    parser = argparse.ArgumentParser(description="有据选购：商品筛选、检索或带引用回答；默认不调用 API。")
    parser.add_argument("query", nargs="?", default="", help="问题；--products 模式下为商品关键词，可留空")
    parser.add_argument("--session-id", help="持久会话 ID；与 --recommend 配合继续追问")
    parser.add_argument("--history", action="store_true", help="查看 --session-id 对应的最近20轮记录")
    parser.add_argument("--fit-review", choices=("semantic", "rules"), default="semantic",
                        help="选购适配审核；semantic逐项审核，rules保留原规则基线")
    parser.add_argument("--recommend", action="store_true", help="自然语言单轮选购：理解需求、筛选、检索、解释")
    parser.add_argument("--products", action="store_true", help="按关键词和明确条件筛选商品，不生成回答")
    parser.add_argument("--category", help="商品筛选：一级品类精确匹配")
    parser.add_argument("--brand", help="商品筛选：品牌精确匹配")
    parser.add_argument("--min-price", help="商品筛选：SKU 最低价")
    parser.add_argument("--max-price", help="商品筛选：SKU 最高价")
    parser.add_argument("--limit", type=int, default=10, help="商品筛选：最多展示 1～50 个商品")
    parser.add_argument("--product-id", help="筛选或检索时限定商品 ID")
    parser.add_argument("--dataset", choices=("original", "products", "all"),
                        help="资料集；关键词默认 all，向量默认 original，保留旧索引")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--method", choices=("keyword", "vector"), default=None)
    parser.add_argument("--online", action="store_true", help="允许缺少缓存时调用 Embedding 或 LLM")
    parser.add_argument("--answer", action="store_true", help="根据检索证据生成回答或复用回答缓存")
    parser.add_argument("--refresh-answer", action="store_true", help="重新生成回答，需同时指定 --answer --online")
    parser.add_argument("--json", action="store_true", help="输出结构化结果")
    args = parser.parse_args()
    args.method = args.method or ("vector" if args.recommend else "keyword")
    if args.session_id and not (args.recommend or args.history):
        parser.error("--session-id 需要 --recommend 或 --history")
    if args.history and (not args.session_id or args.query or args.online or args.products or args.answer):
        parser.error("--history 需要 --session-id，且不与查询、--online、商品筛选或回答生成组合")
    if args.recommend and (args.products or args.answer or args.refresh_answer or args.dataset or args.product_id
                           or args.category or args.brand or args.min_price or args.max_price):
        parser.error("--recommend 从自然语言提取条件，不与手动筛选、资料集或 --answer 参数组合")
    if args.refresh_answer and not (args.answer and args.online):
        parser.error("--refresh-answer 需要 --answer --online，会重新产生 LLM 用量")
    if args.products and (args.answer or args.online or args.dataset or args.method != "keyword"):
        parser.error("--products 是离线商品筛选，不与 --answer、--online、--dataset 或向量模式组合")
    filters = (args.category, args.brand, args.min_price, args.max_price)
    if not args.products and any(value is not None for value in filters):
        parser.error("品类、品牌和价格条件需要 --products；资料问答可用 --product-id 限定商品")
    if not args.products and not args.history and not args.query.strip():
        parser.error("请填写问题，或使用 --products 筛选商品")
    try:
        if args.history:
            snapshot = SessionStore().load(args.session_id, history_limit=20)
            if args.json:
                print(json.dumps(snapshot, ensure_ascii=False, indent=2))
            else:
                print(f"会话：{args.session_id}，已保存 {snapshot['revision']} 轮。")
                for turn in snapshot["turns"]:
                    print(f"\n第 {turn['revision']} 轮：{turn['query']}")
                    print("结果：" + turn["result"]["status"])
                    for index, card in enumerate(turn["result"].get("candidates", []), 1):
                        print(f"  候选 {index}：{card['title']} / {card['product_id']}")
            return 0
        if args.recommend:
            result = (ConversationAssistant(shopping=ShoppingAssistant(review_mode=args.fit_review)).run(
                args.session_id, args.query, online=args.online, method=args.method)
                if args.session_id else ShoppingAssistant(review_mode=args.fit_review).run(
                    args.query, online=args.online, method=args.method))
            if args.json:
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                show_shopping(result)
            return 2 if result["status"] in ("failed", "generation_failed", "generation_disabled", "invalid_answer", "session_conflict") else 0
        if args.products:
            products = search_products(args.query, category=args.category, brand=args.brand,
                                       min_price=args.min_price, max_price=args.max_price,
                                       product_id=args.product_id, limit=args.limit)
            if args.json:
                print(json.dumps(products, ensure_ascii=False, indent=2))
            else:
                show_products(products)
            return 0
        dataset = args.dataset or ("original" if args.method == "vector" else "all")
        chunks, scope = load_dataset(dataset)
        retriever = (KeywordRetriever(chunks) if args.method == "keyword" else
                     VectorRetriever(chunks, EmbeddingClient(load_config()),
                                     index_path=index_path(dataset), online=args.online))
        result = retriever.search(args.query, args.top_k,
                                  product_ids={args.product_id} if args.product_id else None)
        answer = None
        if args.answer:
            answer = GroundedAnswerer().answer(
                args.query, result, online=args.online, refresh=args.refresh_answer, scope=scope)
    except (OSError, ValueError, TypeError, sqlite3.Error, OverflowError) as exc:
        print("处理失败：" + str(exc), file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps({"retrieval": result, "answer": answer} if args.answer else result,
                         ensure_ascii=False, indent=2))
    elif answer is not None:
        show_answer(answer)
        if answer["status"] in ("generation_failed", "invalid_answer", "generation_disabled"):
            print("\n保留本轮检索资料，供核对：")
            show_evidence(result)
    elif result["status"] == "unknown_model":
        print("查询包含未收录的型号编号，未返回其他型号的资料。")
    elif result["status"] == "unknown_product":
        print("指定商品未收录，未返回其他商品的资料。")
    elif result["status"] == "needs_topic":
        print("请在型号后补充要查询的内容，例如内存或处理器。")
    elif not result["hits"]:
        print("未找到问题内容对应的关键词片段；这不证明资料一定没有答案。")
    else:
        show_evidence(result)
        print("\n当前仅展示候选片段；添加 --answer 可生成带引用回答。")
    if not args.json and result.get("embedding_stats") is not None:
        print("Embedding 统计：" + json.dumps(result["embedding_stats"], ensure_ascii=False))
    if answer and answer["status"] in ("generation_failed", "invalid_answer", "generation_disabled"):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
