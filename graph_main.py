"""Optional LangGraph CLI. main.py remains the original implementation."""
import argparse
import json
import sys

from app.graph_workflow import GraphAssistant
from app.shopping import ShoppingAssistant
from main import show_shopping


def main():
    parser = argparse.ArgumentParser(description="Checkpointed shopping workflow")
    parser.add_argument("query", nargs="?")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--method", choices=("keyword", "vector"), default="vector")
    parser.add_argument("--fit-review", choices=("semantic", "rules"), default="semantic")
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--retry", action="store_true", help="Retry the failed node; completed nodes are retained")
    parser.add_argument("--state", action="store_true", help="Inspect without running the graph")
    parser.add_argument("--stream", action="store_true", help="Print node progress on stderr")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.state and (args.query or args.retry):
        parser.error("--state 不能同时提交问题或 --retry")
    assistant = GraphAssistant(shopping=ShoppingAssistant(review_mode=args.fit_review))
    def event(value):
        print(f"[{value['node']}] {value['status']}", file=sys.stderr, flush=True)
    try:
        result = assistant.run(args.session_id, args.query, online=args.online, method=args.method,
            retry=args.retry, inspect_only=args.state, on_event=event if args.stream else None)
    except (ValueError, OSError) as exc:
        parser.exit(2, str(exc) + "\n")
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        show_shopping(result)
        if result.get("paused"):
            print("流程已保存并暂停。用同一 --session-id 提交补充条件即可继续。")
        elif result.get("resumable"):
            print("进度已保存。修复原因后，使用同一 --session-id --retry 重试失败节点。")
    return 1 if result.get("resumable") else 0


if __name__ == "__main__":
    raise SystemExit(main())
