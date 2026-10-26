"""Real-model business dialogue evaluation, cache-first with explicit call limits."""

import argparse
from collections import Counter, defaultdict
from contextlib import closing
import copy
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import time

from app.catalog import DEFAULT_DB, connect_catalog, load_catalog_chunks, money_cents
from app.conversation import ConversationAssistant, DialogueExtractor
from app.corpus import index_path
from app.embedding import EmbeddingClient, load_config
from app.llm import LLMClient, LLMError, load_llm_config
from app.rag import GroundedAnswerer, validate_answer
from app.fit_review import FitReviewer, guarded_checks
from app.retrieval import ROOT, KeywordRetriever
from app.sessions import SessionStore
from app.shopping import NeedsExtractor, ShoppingAssistant, sku_fact
from app.vector_retrieval import VectorRetriever, corpus_digest

DEFAULT_CASES = ROOT / "evaluations/business_cases.jsonl"
EXPECTED_KEYS = {
    "statuses", "requirements", "preference_contains", "min_candidates", "max_candidates",
    "preserve_fields", "max_sku_price", "candidate_brands", "excluded_brands",
    "focus_from_previous", "focus_product_id", "cheaper_than_previous", "exclude_previous",
    "state_options", "unsupported_nonempty", "state_saved", "saved_requirements",
    "cheaper_feasibility_from_previous",
}


class EvaluationBudgetExceeded(RuntimeError):
    pass


class CallBudget:
    def __init__(self, llm_limit, embedding_limit):
        self.limits = {"llm": llm_limit, "embedding": embedding_limit}
        self.counts = {"llm": 0, "embedding": 0}
        self.logs = []

    def take(self, kind):
        if self.counts[kind] >= self.limits[kind]:
            raise EvaluationBudgetExceeded(kind + " 调用上限已到，未发出新请求。")
        self.counts[kind] += 1


class BudgetLLM(LLMClient):
    def __init__(self, config, budget):
        super().__init__(config)
        self.budget = budget

    def chat_json(self, messages, online=False):
        if not online:
            return super().chat_json(messages, online=False)
        self.budget.take("llm")
        started = time.monotonic()
        log = {"kind": "llm", "request": self.budget.counts["llm"]}
        try:
            output = super().chat_json(messages, online=True)
            log.update(raw_output=output, usage={key: value for key, value in self.stats.items() if key != "requests"})
            return output
        except LLMError as exc:
            log["error"] = str(exc)
            raise
        finally:
            log["elapsed_ms"] = round((time.monotonic() - started) * 1000)
            self.budget.logs.append(log)


class BudgetEmbedding(EmbeddingClient):
    def __init__(self, config, budget):
        super().__init__(config)
        self.budget = budget

    def _fetch(self, texts):
        self.budget.take("embedding")
        started = time.monotonic()
        log = {"kind": "embedding", "texts": len(texts), "request": self.budget.counts["embedding"]}
        try:
            return super()._fetch(texts)
        finally:
            log["elapsed_ms"] = round((time.monotonic() - started) * 1000)
            self.budget.logs.append(log)


class RetrieverPool:
    def __init__(self, method, chunks, embedding_config, budget):
        self.method, self.budget, self.embedding_config = method, budget, embedding_config
        self.base = (KeywordRetriever(chunks) if method == "keyword" else
                     VectorRetriever(chunks, BudgetEmbedding(embedding_config, budget),
                                     index_path=index_path("products")))

    def make(self, method, online):
        if method != self.method:
            raise ValueError("评测检索方式不一致。")
        retriever = copy.copy(self.base)
        if method == "vector":
            retriever.client = BudgetEmbedding(self.embedding_config, self.budget)
            retriever.online = online
        return retriever


def load_cases(path=DEFAULT_CASES):
    cases = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not cases or len({case["id"] for case in cases}) != len(cases):
        raise ValueError("业务用例为空或 ID 重复。")
    for case in cases:
        if not isinstance(case.get("turns"), list) or not case["turns"]:
            raise ValueError("每个业务用例必须有对话轮次。")
        for turn in case["turns"]:
            expected = turn.get("expect")
            if not isinstance(turn.get("query"), str) or not turn["query"].strip():
                raise ValueError("评测问题必须是非空文本。")
            if not isinstance(expected, dict) or set(expected) - EXPECTED_KEYS:
                raise ValueError("评测断言字段无效。")
            if not isinstance(expected.get("statuses"), list) or not expected["statuses"]:
                raise ValueError("每轮必须明确预期状态，不能空断言通过。")
            if turn.get("fault") not in (None, "context_network"):
                raise ValueError("未知故障注入类型。")
    return cases


def equal(actual, expected):
    if isinstance(actual, list) and isinstance(expected, list):
        return sorted(actual, key=str) == sorted(expected, key=str)
    return actual == expected


def check_turn(result, expected, previous_cards, first_needs, before, after, chunks_by_id, catalog):
    checks = []

    def check(name, passed, detail=""):
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    def subset(name, actual, values):
        if values is None:
            check(name, actual is None, f"actual={actual!r}")
            return
        for key, value in values.items():
            observed = (actual or {}).get(key)
            check(name + "." + key, equal(observed, value), f"expected={value!r}; actual={observed!r}")

    check("status", result.get("status") in expected["statuses"],
          "expected=" + str(expected["statuses"]) + "; actual=" + str(result.get("status")))
    needs = result.get("requirements")
    cards = result.get("candidates", [])
    if "requirements" in expected:
        subset("requirements", needs, expected["requirements"])
    if "saved_requirements" in expected:
        subset("saved_requirements", after["state"]["requirements"], expected["saved_requirements"])
    for token in expected.get("preference_contains", []):
        check("preference_contains", any(token in value for value in (needs or {}).get("preferences", [])), token)
    for field in expected.get("preserve_fields", []):
        check("preserve." + field, first_needs is not None and
              equal((needs or {}).get(field), first_needs.get(field)), field)
    if "min_candidates" in expected:
        check("min_candidates", len(cards) >= expected["min_candidates"], str(len(cards)))
    if "max_candidates" in expected:
        check("max_candidates", len(cards) <= expected["max_candidates"], str(len(cards)))
    if expected.get("unsupported_nonempty"):
        check("unsupported_constraint_visible", bool((needs or {}).get("unsupported_constraints")))
    if "state_options" in expected:
        subset("state_options", after["state"]["search_options"], expected["state_options"])
    saved = expected.get("state_saved", True)
    check("state_saved", result.get("state_saved") is saved)
    if saved:
        check("revision_increment", after["revision"] == before["revision"] + 1)
        if needs is not None:
            check("saved_needs_match", after["state"]["requirements"] == needs)
    else:
        check("failure_preserves_state", after == before)
    target = None
    for field in ("focus_from_previous", "cheaper_than_previous", "exclude_previous"):
        if field in expected:
            index = expected[field] - 1
            if not 0 <= index < len(previous_cards):
                check(field, False, "前轮没有相应候选")
                continue
            target = previous_cards[index]
            if field == "focus_from_previous":
                check(field, result.get("focus_product_id") == target["product_id"])
                check("focused_candidates_only", bool(cards) and all(
                    card["product_id"] == target["product_id"] for card in cards))
            elif field == "cheaper_than_previous":
                ceiling = (money_cents(target["sku"]["price"]) - 1) / 100
                check("relative_price_ceiling", result.get("relative_price_ceiling") == ceiling)
                check(field, all(card["sku"]["price"] < target["sku"]["price"] for card in cards))
            else:
                check(field, all(card["product_id"] != target["product_id"] for card in cards))
    if "cheaper_feasibility_from_previous" in expected:
        index = expected["cheaper_feasibility_from_previous"] - 1
        if not 0 <= index < len(previous_cards) or not needs:
            check("alternative_availability", False, "前轮候选或需求缺失")
        else:
            reference = previous_cards[index]
            prior_options = before.get("state", {}).get("search_options", {})
            excluded = set(prior_options.get("excluded_product_ids", [])) | {reference["product_id"]}
            ceiling = money_cents(reference["sku"]["price"]) - 1
            if prior_options.get("price_ceiling") is not None:
                ceiling = min(ceiling, money_cents(prior_options["price_ceiling"]))
            if needs["max_price"] is not None:
                ceiling = min(ceiling, money_cents(needs["max_price"]))
            floor = money_cents(needs["min_price"]) if needs["min_price"] is not None else 0
            eligible = {sku["product_id"] for sku in catalog.values()
                        if sku["product_id"] not in excluded and floor <= sku["price_cents"] <= ceiling
                        and (not needs["sub_categories"] or sku["sub_category"] in needs["sub_categories"])
                        and (not needs["brand"] or sku["brand"] == needs["brand"])
                        and sku["brand"] not in needs["excluded_brands"]}
            if eligible:
                check("alternative_availability", bool(cards) and result.get("status") in
                      ("completed", "partial", "insufficient_evidence"),
                      "独立商品表检查存在可用替代：" + str(sorted(eligible)))
                check("alternative_membership", all(card["product_id"] in eligible for card in cards))
            else:
                check("alternative_availability", result.get("status") == "no_products" and not cards,
                      "独立商品表检查没有更低价且满足原硬条件的替代")
    if "focus_product_id" in expected:
        check("focus_product_id", result.get("focus_product_id") == expected["focus_product_id"])
        check("focused_candidates_only", bool(cards) and all(
            card["product_id"] == expected["focus_product_id"] for card in cards))
    for card in cards:
        sku, product_id = card["sku"], card["product_id"]
        authority = catalog.get(sku["sku_id"])
        check("catalog_sku_identity", authority is not None and authority["product_id"] == product_id)
        if authority:
            check("catalog_price", money_cents(sku["price"]) == authority["price_cents"])
            check("catalog_properties", sku["properties"] == json.loads(authority["properties_json"]))
        if needs:
            if needs["max_price"] is not None:
                check("hard_max_price", sku["price"] <= needs["max_price"])
            if needs["min_price"] is not None:
                check("hard_min_price", sku["price"] >= needs["min_price"])
            if needs["sub_categories"]:
                check("hard_product_type", card["sub_category"] in needs["sub_categories"])
            if needs["brand"]:
                check("hard_brand", card["brand"] == needs["brand"])
            check("hard_excluded_brand", card["brand"] not in needs["excluded_brands"])
        if "max_sku_price" in expected:
            check("gold_max_sku_price", sku["price"] <= expected["max_sku_price"])
        if "candidate_brands" in expected:
            check("gold_candidate_brand", card["brand"] in expected["candidate_brands"])
        check("gold_excluded_brand", card["brand"] not in expected.get("excluded_brands", []))
    answer = result.get("answer") or {}
    if answer.get("status") in ("answered", "partial", "insufficient_evidence"):
        try:
            evidence = {**chunks_by_id}
            for card in cards:
                evidence_row = sku_fact({**card, "local_path": catalog[card["sku"]["sku_id"]]["local_path"]}, card["sku"])
                evidence[evidence_row["chunk_id"]] = evidence_row
            retrieved = [evidence[key] for key in answer["retrieved_chunk_ids"]]
            scope = {"shopping_allowed_products": [card["product_id"] for card in cards]}
            if result.get("focus_product_id"):
                scope["shopping_focus_product"] = result["focus_product_id"]
            if result.get("fit_review", {}).get("status") == "reviewed":
                tasks = [{"product_id": card["product_id"], "requirement": preference}
                         for card in cards for preference in needs["preferences"]]
                rechecked = guarded_checks({"checks": result["fit_review"]["model_checks"]}, tasks, cards, retrieved)
                check("fit_review_matrix_revalidated", rechecked == result["fit_review"]["checks"])
            else:
                validate_answer({key: answer[key] for key in ("status", "claims", "missing_information")},
                                retrieved, scope)
            check("citation_structure_and_attribution", True)
        except (ValueError, KeyError, TypeError) as exc:
            check("citation_structure_and_attribution", False, str(exc))
    return checks


def summarize(rows):
    counts = Counter(row["outcome"] for row in rows)
    executed = counts["passed"] + counts["failed"]
    groups = {}
    for category in sorted({row["category"] for row in rows}):
        matching = [row for row in rows if row["category"] == category]
        groups[category] = dict(Counter(row["outcome"] for row in matching))
    return {
        "total_cases": len(rows), "passed": counts["passed"], "failed": counts["failed"],
        "not_run_cache": counts["not_run_cache"], "not_run_budget": counts["not_run_budget"],
        "executed_cases": executed, "execution_coverage": executed / len(rows) if rows else 0,
        "pass_rate_executed": counts["passed"] / executed if executed else None,
        "passed_over_all_cases": counts["passed"] / len(rows) if rows else 0,
        "by_category": groups,
    }


def run_suite(cases, mode="cached", method="vector", max_llm_calls=60, max_embedding_calls=30, fit_review_mode="rules"):
    if mode not in ("cached", "online") or method not in ("keyword", "vector"):
        raise ValueError("未知评测模式或检索方式。")
    if max_llm_calls < 0 or max_embedding_calls < 0:
        raise ValueError("请求上限不能为负。")
    online = mode == "online"
    budget = CallBudget(max_llm_calls if online else 0, max_embedding_calls if online else 0)
    config = load_llm_config()
    embedding_config = load_config() if method == "vector" else None
    chunks = load_catalog_chunks()
    pool = RetrieverPool(method, chunks, embedding_config, budget)
    chunk_map = {row["chunk_id"]: row for row in chunks}
    with closing(connect_catalog()) as connection:
        catalog = {row["sku_id"]: dict(row) for row in connection.execute("SELECT s.*, p.local_path, p.brand, p.sub_category FROM skus s JOIN products p ON p.product_id=s.product_id")}
    rows = []
    with TemporaryDirectory(prefix="shopping-business-evals-") as directory:
        store = SessionStore(Path(directory) / "sessions.sqlite")
        for case in cases:
            row = {"id": case["id"], "title": case["title"], "category": case["category"],
                   "outcome": "passed", "turns": []}
            first_needs, last_shown = {}, {}
            for number, turn in enumerate(case["turns"], 1):
                session = case["id"] + "_" + turn.get("session", "main")
                before = store.load(session)
                shopping = ShoppingAssistant(
                    extractor=NeedsExtractor(BudgetLLM(config, budget)),
                    answerer=GroundedAnswerer(BudgetLLM(config, budget)),
                    retriever_factory=pool.make, review_mode=fit_review_mode,
                    reviewer=FitReviewer(BudgetLLM(replace(config, max_tokens=4000), budget)))
                dialogue = DialogueExtractor(BudgetLLM(config, budget))
                if turn.get("fault") == "context_network":
                    class FaultDialogue:
                        def extract(self, *args, **kwargs):
                            raise LLMError("评测注入：上下文解析网络失败。")
                    dialogue = FaultDialogue()
                assistant = ConversationAssistant(store, shopping, dialogue)
                started = time.monotonic()
                calls_before = dict(budget.counts)
                try:
                    result = assistant.run(session, turn["query"], online=online, method=method)
                    after = SessionStore(store.path).load(session)
                    record = {"turn": number, "query": turn["query"], "session": session,
                              "fault_injected": turn.get("fault"), "result": result,
                              "expect": turn["expect"]}
                    if mode == "cached" and (
                        result["status"] == "generation_disabled" or
                        "缺少向量缓存" in result.get("error", "")
                    ):
                        row["outcome"] = "not_run_cache"
                        record["checks"] = []
                    else:
                        if session not in first_needs and result.get("requirements") is not None:
                            first_needs[session] = copy.deepcopy(result["requirements"])
                        record["checks"] = check_turn(
                            result, turn["expect"], last_shown.get(session, []),
                            first_needs.get(session), before, after, chunk_map, catalog)
                        if not all(check["passed"] for check in record["checks"]):
                            row["outcome"] = "failed"
                    if result.get("state_saved") and result.get("candidates"):
                        last_shown[session] = copy.deepcopy(result["candidates"])
                except EvaluationBudgetExceeded as exc:
                    row["outcome"] = "not_run_budget"
                    record = {"turn": number, "query": turn["query"], "checks": [], "error": str(exc)}
                except Exception as exc:
                    row["outcome"] = "failed"
                    record = {"turn": number, "query": turn["query"], "checks": [],
                              "runner_error": type(exc).__name__ + ": " + str(exc)}
                record["elapsed_ms"] = round((time.monotonic() - started) * 1000)
                record["new_api_calls"] = {kind: budget.counts[kind] - calls_before[kind] for kind in budget.counts}
                row["turns"].append(record)
                if row["outcome"] != "passed":
                    row["unexecuted_followup_turns"] = len(case["turns"]) - number
                    break
            rows.append(row)
            print(f"{case['id']} {row['outcome']} | {case['title']} | calls={budget.counts}", flush=True)
    code_paths = sorted((ROOT / "app").glob("*.py")) + [Path(__file__)]
    return {
        "evaluation_kind": "real_model_business_dialogues", "mode": mode, "method": method, "fit_review_mode": fit_review_mode,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "llm_profile": config.profile,
        "fit_review_profile": replace(config, max_tokens=4000).profile if fit_review_mode == "semantic" else None,
        "embedding_profile": embedding_config.profile if embedding_config else None,
        "corpus_sha256": corpus_digest(chunks),
        "catalog_file_sha256": hashlib.sha256(DEFAULT_DB.read_bytes()).hexdigest(),
        "code_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in code_paths},
        "new_api_calls": budget.counts, "api_limits": budget.limits, "api_logs": budget.logs,
        "limitations": [
            "人工编写的开发对话集，部分起始问题共用同一缓存，不是独立最终测试集。",
            "正常对话使用实际模型输出或实际模型缓存；网络恢复用例明确注入故障。",
            "通过率衡量标注的状态、条件、指代、SKU 和引用约束，不等于推荐准确率。",
            "相似度不是置信度；引用存在不证明完整语义蕴含。",
            "缓存缺失和达到调用上限单列，不计作已通过；每个场景首个失败后停止后续轮次。",
            "会话使用临时数据库，不修改用户已有会话。没有模拟模型答案冒充真实质量评测。",
        ],
        "summary": summarize(rows), "results": rows,
    }


def review_markdown(report):
    summary = report["summary"]
    lines = ["# 业务对话评测", "",
             f"场景：{summary['total_cases']}；通过：{summary['passed']}；失败：{summary['failed']}；"
             f"缓存未就绪：{summary['not_run_cache']}；请求预算耗尽：{summary['not_run_budget']}。",
             "", "这些结果衡量预定义业务约束，不代表推荐准确率。", "", "## 失败或未执行", ""]
    for row in report["results"]:
        if row["outcome"] == "passed":
            continue
        lines += [f"### {row['id']} {row['title']}：{row['outcome']}", ""]
        for turn in row["turns"]:
            failed = [check for check in turn["checks"] if not check["passed"]]
            if failed or turn.get("error") or turn.get("runner_error") or row["outcome"].startswith("not_run"):
                lines.append("用户：" + turn["query"])
                result = turn.get("result", {})
                lines.append("实际状态：" + str(result.get("status", "")))
                if result.get("error"):
                    lines.append("错误：" + result["error"])
                for check in failed:
                    lines.append("- " + check["name"] + "：" + check["detail"])
                if turn.get("error") or turn.get("runner_error"):
                    lines.append(str(turn.get("error") or turn.get("runner_error")))
                lines.append("")
    lines += ["## 推荐理由人工复核清单", "",
              "逐条核对：是否真正回答需求、偏好是否有证据、是否遗漏负面取舍、是否把个别评论当总体事实。"
              "下列项目尚未自动判定语义正确。", ""]
    seen = set()
    for row in report["results"]:
        for turn in row["turns"]:
            result = turn.get("result", {})
            answer = result.get("answer") or {}
            claims = answer.get("claims", [])
            signature = json.dumps(claims, ensure_ascii=False, sort_keys=True)
            if not claims or signature in seen:
                continue
            seen.add(signature)
            lines += [f"### {row['id']} 第{turn['turn']}轮", "",
                      "用户：" + turn["query"], "- [ ] 理由与证据一致", "- [ ] 关键取舍已说明", ""]
            for claim in claims:
                lines.append("- 结论：" + claim["text"])
                for citation in claim["citations"]:
                    lines.append("  - 摘录：" + citation["quote"] + "（" + citation["chunk_id"] + "）")
            for missing in answer.get("missing_information", []):
                lines.append("- 尚缺：" + missing)
            lines.append("")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="真实业务多轮评测；默认只复用缓存。")
    parser.add_argument("--mode", choices=("cached", "online"), default="cached")
    parser.add_argument("--method", choices=("keyword", "vector"), default="vector")
    parser.add_argument("--fit-review", choices=("rules", "semantic"), default="rules")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--case-id", action="append", help="只运行指定用例，可重复")
    parser.add_argument("--max-llm-calls", type=int, default=60)
    parser.add_argument("--max-embedding-calls", type=int, default=30)
    args = parser.parse_args()
    cases = load_cases(args.cases)
    if args.case_id:
        selected = set(args.case_id)
        if selected - {case["id"] for case in cases}:
            parser.error("指定的用例 ID 不存在")
        cases = [case for case in cases if case["id"] in selected]
    report = run_suite(cases, args.mode, args.method, args.max_llm_calls, args.max_embedding_calls, args.fit_review)
    report["cases_sha256"] = hashlib.sha256(args.cases.read_bytes()).hexdigest()
    report["selected_case_ids"] = [case["id"] for case in cases]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = ROOT / "runs" / ("business_" + args.mode + "_" + stamp + ".json")
    review = output.with_suffix(".md")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    review.write_text(review_markdown(report), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print("新增 API：" + json.dumps(report["new_api_calls"], ensure_ascii=False))
    print("结果：" + str(output))
    print("人工复核：" + str(review))
    return 1 if report["summary"]["failed"] else (2 if report["summary"]["executed_cases"] < len(cases) else 0)


if __name__ == "__main__":
    raise SystemExit(main())
