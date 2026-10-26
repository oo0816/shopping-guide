"""Compare rule baseline and real model+guard reviews on controlled semantic cases."""

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from app.fit_review import FitReviewer
from app.llm import load_llm_config
from app.retrieval import ROOT
from app.selection_policy import assess_suitability
from evaluations.run_business_evals import BudgetLLM, CallBudget, EvaluationBudgetExceeded

DEFAULT_CASES = ROOT / "evaluations/fit_review_cases.jsonl"


def evaluate(cases, reviewer, online=False):
    rows = []
    for case in cases:
        baseline = assess_suitability({"preferences": [case["requirement"]]}, case["evidence"])
        rule_status = {"supported": "supported", "caution": "limited", "unknown": "unknown"}[baseline["status"]]
        row = {"id": case["id"], "title": case["title"], "requirement": case["requirement"],
               "expected_status": case["expected_status"], "rule_status": rule_status,
               "rule_correct": rule_status == case["expected_status"], "data_kind": case["data_kind"]}
        try:
            review = reviewer.review({"preferences": [case["requirement"]]}, case["cards"],
                                     case["evidence"], online=online)
        except EvaluationBudgetExceeded as exc:
            row.update(outcome="not_run_budget", error=str(exc))
            rows.append(row)
            continue
        row["review"] = review
        if review["status"] == "review_disabled":
            row["outcome"] = "not_run_cache"
        elif review["status"] != "reviewed":
            row["outcome"] = "failed"
        else:
            check = review["checks"][0]
            row["model_status"] = review["model_checks"][0]["status"]
            row["model_correct"] = row["model_status"] == case["expected_status"]
            row["guarded_status"] = check["status"]
            row["status_correct"] = check["status"] == case["expected_status"]
            row["condition_preserved"] = (not case.get("condition_contains") or any(
                case["condition_contains"] in condition for condition in check["conditions"]))
            row["outcome"] = "passed" if row["status_correct"] and row["condition_preserved"] else "failed"
        rows.append(row)
        print(case["id"] + " " + row["outcome"] + " | rule=" + rule_status +
              " | review=" + str(row.get("guarded_status", review["status"])), flush=True)
    executed = [row for row in rows if row["outcome"] not in ("not_run_cache", "not_run_budget")]
    return {
        "total": len(rows), "executed": len(executed),
        "passed": sum(row["outcome"] == "passed" for row in rows),
        "failed": sum(row["outcome"] == "failed" for row in rows),
        "not_run": len(rows) - len(executed),
        "rule_status_correct": sum(row["rule_correct"] for row in executed),
        "raw_model_status_correct": sum(row.get("model_correct", False) for row in executed),
        "guarded_status_correct": sum(row.get("status_correct", False) for row in executed),
        "results": rows,
    }


def main():
    parser = argparse.ArgumentParser(description="语义适配专项对照评测，默认只读真实模型缓存。")
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--max-llm-calls", type=int, default=14)
    args = parser.parse_args()
    cases = [json.loads(line) for line in DEFAULT_CASES.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not cases or len({case["id"] for case in cases}) != len(cases):
        parser.error("专项用例为空或重复")
    budget = CallBudget(args.max_llm_calls if args.online else 0, 0)
    config = replace(load_llm_config(), max_tokens=4000)
    reviewer = FitReviewer(BudgetLLM(config, budget))
    result = evaluate(cases, reviewer, online=args.online)
    report = {
        "kind": "controlled_semantic_fit_review", "created_at": datetime.now(timezone.utc).isoformat(),
        "model_profile": config.profile,
        "cases_sha256": hashlib.sha256(DEFAULT_CASES.read_bytes()).hexdigest(),
        "code_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                        for name in ("app/fit_review.py", "app/selection_policy.py")},
        "new_api_calls": budget.counts, "api_logs": budget.logs,
        "limitations": [
            "人工构造的14条对照证据，不是真实商品新增事实，也不是独立最终测试集。",
            "规则和模型读取同一份证据。模型为实际调用或其实际缓存；最终状态包含确定性范围保护。",
            "分开报告原始模型状态和程序保护后的状态，不能把程序纠正算成模型自己答对。",
            "只验证标注的状态和限定条件保留，不证明理由完整语义正确或真实选购准确率。",
        ],
        **result,
    }
    output = ROOT / "runs" / ("fit_semantic_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "results"}, ensure_ascii=False, indent=2))
    print("新增调用：" + json.dumps(budget.counts, ensure_ascii=False))
    print("报告：" + str(output))
    return 1 if result["failed"] else (2 if result["not_run"] else 0)


if __name__ == "__main__":
    raise SystemExit(main())
