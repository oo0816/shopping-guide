"""Cached parity evaluation on actual business cases; online APIs are prohibited."""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

from app.catalog import load_catalog_chunks
from app.conversation import ConversationAssistant, DialogueExtractor
from app.embedding import load_config
from app.fit_review import FitReviewer
from app.graph_workflow import GraphAssistant
from app.llm import load_llm_config
from app.rag import GroundedAnswerer
from app.retrieval import ROOT
from app.sessions import SessionStore
from app.shopping import NeedsExtractor, ShoppingAssistant
from evaluations.run_business_evals import BudgetLLM, CallBudget, DEFAULT_CASES, RetrieverPool, load_cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", action="append", help="Default: B01, B18; no online evaluation")
    parser.add_argument("--fit-review", choices=("semantic", "rules"), default="semantic")
    args = parser.parse_args()
    selected = set(args.case_id or ("B01", "B18"))
    cases = [case for case in load_cases(DEFAULT_CASES) if case["id"] in selected]
    if {case["id"] for case in cases} != selected:
        parser.error("用例 ID 不存在")
    budget = CallBudget(0, 0)
    config = load_llm_config()
    pool = RetrieverPool("vector", load_catalog_chunks(), load_config(), budget)
    def shopping():
        return ShoppingAssistant(extractor=NeedsExtractor(BudgetLLM(config, budget)),
            answerer=GroundedAnswerer(BudgetLLM(config, budget)), retriever_factory=pool.make,
            reviewer=FitReviewer(BudgetLLM(replace(config, max_tokens=4000), budget)), review_mode=args.fit_review)
    rows = []
    with TemporaryDirectory(prefix="shopping-graph-parity-") as directory:
        for case in cases:
            store = SessionStore(Path(directory) / (case["id"] + "-old.sqlite"))
            row = {"id": case["id"], "outcome": "passed", "turns": []}
            for number, turn in enumerate(case["turns"], 1):
                if turn.get("fault"):
                    row["outcome"] = "not_run_fault_case"
                    break
                session = case["id"] + "_" + turn.get("session", "main")
                old = ConversationAssistant(store, shopping(), DialogueExtractor(BudgetLLM(config, budget))).run(session, turn["query"], online=False)
                graph = GraphAssistant(Path(directory) / (case["id"] + "-graph.sqlite"), shopping(), DialogueExtractor(BudgetLLM(config, budget)))
                new = graph.run(session, turn["query"], online=False)
                if "generation_disabled" in (old["status"], new["status"]):
                    row["outcome"] = "not_run_cache"
                    row["turns"].append({"turn": number, "old": old, "graph": new})
                    break
                keys = ("status", "requirements", "candidates", "answer", "questions", "unsupported_constraints", "relative_price_ceiling", "focus_product_id")
                differences = [key for key in keys if old.get(key) != new.get(key)]
                if differences:
                    row["outcome"] = "failed"
                row["turns"].append({"turn": number, "differences": differences, "old": old, "graph": new})
                if differences or new.get("resumable"):
                    break
            rows.append(row)
    report = {"engine": "langgraph", "mode": "cached_parity", "fit_review": args.fit_review,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "versions": {name: version(name) for name in ("langgraph", "langgraph-checkpoint-sqlite")},
        "code_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
                         ("app/graph_workflow.py", "app/shopping.py", "app/fit_review.py", "app/conversation.py")},
        "cases_sha256": hashlib.sha256(DEFAULT_CASES.read_bytes()).hexdigest(),
        "summary": {key: sum(row["outcome"] == key for row in rows) for key in ("passed", "failed", "not_run_cache", "not_run_fault_case")},
        "new_api_calls": budget.counts, "results": rows}
    output = ROOT / "runs" / ("graph_cached_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": report["summary"], "new_api_calls": budget.counts, "report": str(output)}, ensure_ascii=False))
    return 1 if report["summary"]["failed"] else (2 if report["summary"]["not_run_cache"] or report["summary"]["not_run_fault_case"] else 0)


if __name__ == "__main__":
    raise SystemExit(main())
