"""Parallel, checkpointed workflow; the original CLI and session database stay independent."""
import copy
import hashlib
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, RetryPolicy, interrupt

from app.conversation import DialogueExtractor, ReferenceClarification
from app.fit_review import FitReviewer, apply_review, review_answer
from app.llm import LLMError
from app.retrieval import ROOT
from app.sessions import SessionStore, empty_state
from app.shopping import OnlineRequired, ShoppingAssistant, catalog_context


class ShoppingState(TypedDict, total=False):
    query: str
    method: str
    review_mode: str
    memory: dict
    turns: list
    result: dict
    needs: dict
    effective_query: str
    options: dict
    focus_id: str | None
    detail: bool
    products: list
    hits: list
    review: dict
    revision: int


class StageFailure(Exception):
    def __init__(self, stage, status, message):
        self.stage, self.status, self.message = stage, status, message
        super().__init__(message)


def transient_local_error(exc):
    # Automatic retries are limited to SQLite contention, never billable API calls.
    return isinstance(exc, sqlite3.OperationalError) and any(
        word in str(exc).lower() for word in ("locked", "busy"))


def guarded(stage, function):
    def node(state: ShoppingState, config: RunnableConfig):
        try:
            return function(state, config)
        except StageFailure:
            raise
        except OnlineRequired as exc:
            raise StageFailure(stage, "generation_disabled", str(exc)) from None
        except LLMError as exc:
            raise StageFailure(stage, "generation_failed", str(exc)) from None
        except (OSError, ValueError, TypeError, sqlite3.Error) as exc:
            if stage == "product_search" and transient_local_error(exc):
                raise
            raise StageFailure(stage, "failed", str(exc)) from None
    return node


class ShoppingWorkflow:
    def __init__(self, checkpointer, shopping=None, dialogue=None):
        self.shopping = shopping or ShoppingAssistant()
        self.dialogue = dialogue or DialogueExtractor()
        builder = StateGraph(ShoppingState)
        for name, function in (
            ("requirements", self.requirements), ("product_search", self.product_search),
            ("evidence_retrieval", self.evidence_retrieval), ("fit_review", self.fit_review),
            ("recommendation", self.recommendation), ("commit_turn", self.commit_turn),
        ):
            policy = RetryPolicy(max_attempts=2, initial_interval=0.05, jitter=False,
                                 retry_on=transient_local_error) if name == "product_search" else None
            builder.add_node(name, guarded(name, function), retry_policy=policy)
        # Do not catch interrupt's control-flow exception or put API calls before it.
        builder.add_node("human_input", self.human_input)
        builder.add_edge(START, "requirements")
        builder.add_conditional_edges("requirements", lambda s: "human_input" if s["result"]["status"] == "needs_clarification" else "product_search")
        builder.add_conditional_edges("product_search", lambda s: "human_input" if not s["products"] else "evidence_retrieval")
        builder.add_conditional_edges("evidence_retrieval", lambda s: "fit_review" if s["review_mode"] == "semantic" and s["needs"]["preferences"] and not s["detail"] else "recommendation")
        builder.add_edge("fit_review", "recommendation")
        builder.add_edge("recommendation", "commit_turn")
        builder.add_edge("human_input", "requirements")
        builder.add_edge("commit_turn", END)
        self.graph = builder.compile(checkpointer=checkpointer)

    def requirements(self, state, config):
        query = state["query"]
        memory = copy.deepcopy(state.get("memory", empty_state()))
        result = {"query": query, "status": "started", "steps": [], "candidates": [],
                  "requirements": memory["requirements"], "answer": None, "embedding_stats": {"requests": 0}}
        options = copy.deepcopy(memory["search_options"])
        detail, focus_id, effective_query = False, None, query
        online = config["configurable"].get("online", False)
        try:
            if memory["requirements"] is None:
                if re.search(r"第\s*[\d一二三四五六七八九十]", query):
                    raise ReferenceClarification("当前会话还没有候选，请先说明要选购什么商品。")
                needs, stats = self.shopping.extractor.extract(query, catalog_context(self.shopping.db_path), online=online)
            else:
                snapshot = {"state": memory, "turns": state.get("turns", [])}
                resolved, stats = self.dialogue.extract(query, snapshot, catalog_context(self.shopping.db_path), online=online)
                needs, options, focus = resolved["needs"], resolved["options"], resolved["focus"]
                detail = resolved["detail"]
                effective_query = (
                    "当前选购类型：" + "、".join(needs["sub_categories"]) + " " + needs["keywords"]
                    + "；用途和偏好：" + "、".join(needs["preferences"])
                    + ("；正在讨论：" + focus["title"] if focus else "")
                    + "；本轮用户输入：" + query)
                result["effective_query"] = effective_query
                result["steps"].append({"stage": "context", "status": "completed", **stats})
                stats = {"provided_by_context": True, "requests": 0}
                if resolved["new_search"]:
                    memory = empty_state()
                memory["search_options"] = options
                if focus:
                    memory["focus_product"] = focus
                    focus_id = focus["product_id"] if detail else None
            result["requirements"] = needs
            result["steps"].append({"stage": "requirements", "status": "completed", **stats})
            if options["price_ceiling"] is not None:
                result["relative_price_ceiling"] = options["price_ceiling"]
            if focus_id:
                result["focus_product_id"] = focus_id
            if needs["clarification_questions"] or needs["unsupported_constraints"]:
                result.update(status="needs_clarification", questions=needs["clarification_questions"],
                              unsupported_constraints=needs["unsupported_constraints"])
        except ReferenceClarification as exc:
            needs = memory["requirements"]
            result.update(status="needs_clarification", questions=[str(exc)])
        return {"needs": needs, "memory": memory, "options": options, "detail": detail,
                "focus_id": focus_id, "effective_query": effective_query, "result": result,
                "products": [], "hits": [], "review": {}}

    def product_search(self, state, config):
        products, truncated = self.shopping.find_candidates(state["needs"], self.shopping.db_path,
            price_ceiling=state["options"]["price_ceiling"],
            excluded_product_ids=state["options"]["excluded_product_ids"], focus_product_id=state["focus_id"])
        result = copy.deepcopy(state["result"])
        result["steps"].append({"stage": "product_search", "status": "completed", "matched_products": len(products), "search_truncated": truncated})
        if not products:
            result.update(status="no_products", message="当前商品库没有符合这些条件的商品；未放宽预算、品牌或商品类型。")
        return {"products": products, "result": result}

    def evidence_retrieval(self, state, config):
        cards, hits, stats = self.shopping.retrieve_candidates(state["needs"], state["products"], state["effective_query"],
            method=state["method"], online=config["configurable"].get("online", False), detail=state["detail"])
        result = copy.deepcopy(state["result"])
        result.update(candidates=cards, embedding_stats=stats)
        result["steps"].append({"stage": "evidence_retrieval", "status": "completed", "candidate_count": len(cards), "passages": len(hits), "method": state["method"]})
        return {"hits": hits, "result": result}

    def fit_review(self, state, config):
        reviewer = self.shopping.reviewer or FitReviewer()
        review = reviewer.review(state["needs"], state["result"]["candidates"], state["hits"], online=config["configurable"].get("online", False))
        if review["status"] != "reviewed":
            status = {"review_disabled": "generation_disabled", "review_failed": "generation_failed"}.get(review["status"], "invalid_answer")
            raise StageFailure("fit_review", status, review.get("error", "适配审核未完成。"))
        result = copy.deepcopy(state["result"])
        result.update(fit_review=review, rule_candidate_order=[c["product_id"] for c in result["candidates"]], review_pending=False)
        result["candidates"] = apply_review(result["candidates"], review)
        result["steps"].append({"stage": "fit_review", "status": review["status"], "cache_hit": review.get("cache_hit", False), "requests": review.get("llm_stats", {}).get("requests", 0)})
        return {"review": review, "result": result}

    def recommendation(self, state, config):
        if state["review"]:
            answer = review_answer(state["review"], state["hits"])
            step = {"stage": "recommendation", "status": answer["status"], "generated_from": "validated_fit_matrix", "requests": 0}
        else:
            answer = self.shopping.answer_candidates(state["needs"], state["result"]["candidates"], state["hits"], state["effective_query"],
                method=state["method"], online=config["configurable"].get("online", False), detail=state["detail"], focus_product_id=state["focus_id"])
            step = {"stage": "recommendation", "status": answer["status"], "cache_hit": answer.get("cache_hit", False), "requests": answer.get("llm_stats", {}).get("requests", 0)}
        if answer["status"] not in ("answered", "partial", "insufficient_evidence"):
            raise StageFailure("recommendation", answer["status"], answer.get("error", "回答未通过校验。"))
        result = copy.deepcopy(state["result"])
        result.update(answer=answer, status={"answered": "completed", "partial": "partial", "insufficient_evidence": "insufficient_evidence"}[answer["status"]])
        result["steps"].append(step)
        return {"result": result}

    @staticmethod
    def commit_turn(state, config):
        memory = copy.deepcopy(state["memory"])
        result = copy.deepcopy(state["result"])
        if result.get("requirements") is not None:
            memory["requirements"] = result["requirements"]
        if result.get("candidates"):
            memory["last_candidates"] = result["candidates"]
        revision = state.get("revision", 0) + 1
        result.update(session_id=config["configurable"]["thread_id"], turn=revision, state_saved=True)
        turns = (state.get("turns", []) + [{"revision": revision, "query": state["query"], "result": result}])[-6:]
        return {"memory": memory, "revision": revision, "turns": turns, "result": result}

    def human_input(self, state: ShoppingState, config: RunnableConfig):
        response = interrupt({"status": state["result"]["status"],
            "questions": state["result"].get("questions", []),
            "unsupported_constraints": state["result"].get("unsupported_constraints", []),
            "message": state["result"].get("message"), "requirements": state["needs"]})
        if not isinstance(response, str) or not response.strip() or len(response) > 2000:
            raise ValueError("补充条件应为 1～2000 字的文本。")
        committed = self.commit_turn(state, config)
        # Record the pending business turn in the same graph transaction, not another DB.
        return {**committed, "query": response}


@contextmanager
def thread_lock(path, session_id):
    directory = Path(path).parent / "graph_locks"
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / (hashlib.sha256(session_id.encode()).hexdigest() + ".lock")
    with lock_path.open("a+b") as handle:
        if lock_path.stat().st_size == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        import os
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                raise ValueError("该图会话正在运行，请稍后再试。") from None
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


class GraphAssistant:
    def __init__(self, db_path=None, shopping=None, dialogue=None):
        self.db_path = Path(db_path or ROOT / "data/processed/shopping_graph_checkpoints.sqlite")
        self.shopping, self.dialogue = shopping, dialogue

    def run(self, session_id, query=None, *, online=False, method="vector", retry=False, on_event=None, inspect_only=False, include_history=False):
        SessionStore.validate_id(session_id)
        if method not in ("keyword", "vector"):
            raise ValueError("未知检索方式。")
        if retry and query is not None:
            raise ValueError("失败重试不能同时提交新问题。")
        if not retry and not inspect_only and (not isinstance(query, str) or not query.strip() or len(query) > 2000):
            raise ValueError("本轮输入应为 1～2000 字的文本。")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with thread_lock(self.db_path, session_id), SqliteSaver.from_conn_string(str(self.db_path)) as saver:
            graph = ShoppingWorkflow(saver, self.shopping, self.dialogue).graph
            config = {"configurable": {"thread_id": session_id, "online": online}, "recursion_limit": 100}
            before = graph.get_state(config)
            if inspect_only:
                result = self.summary(before, session_id)
                if include_history:
                    result["history"] = copy.deepcopy(before.values.get("turns", []))
                return result
            paused = any(task.interrupts for task in before.tasks)
            if retry:
                if not before.next or paused:
                    raise ValueError("没有失败步骤可重试；等待补充时请直接提交文本。")
                payload = None
            elif paused:
                payload = Command(resume=query)
            elif before.next:
                raise ValueError("该会话有未完成步骤，请先使用 --retry；新选购可换一个会话 ID。")
            else:
                payload = {"query": query, "method": method, "review_mode": (self.shopping.review_mode if self.shopping else "semantic")}
            try:
                for update in graph.stream(payload, config, stream_mode="updates"):
                    if on_event:
                        for name in update:
                            on_event({"node": name, "status": "paused" if name == "__interrupt__" else "completed"})
            except StageFailure as exc:
                result = self.summary(graph.get_state(config), session_id)
                result.update(status=exc.status, failed_stage=exc.stage, error=exc.message, resumable=True)
                return result
            except (OSError, sqlite3.Error) as exc:
                result = self.summary(graph.get_state(config), session_id)
                result.update(status="failed", error=str(exc), resumable=True)
                return result
            return self.summary(graph.get_state(config), session_id)

    @staticmethod
    def summary(snapshot, session_id):
        result = copy.deepcopy(snapshot.values.get("result", {}))
        if snapshot.values.get("query") != result.get("query"):
            result = {"query": snapshot.values.get("query"), "status": "started", "candidates": [], "answer": None,
                      "requirements": snapshot.values.get("memory", {}).get("requirements"), "steps": []}
        interrupts = [item.value for task in snapshot.tasks for item in task.interrupts]
        result.update(session_id=session_id, engine="langgraph", pending_nodes=list(snapshot.next),
                      paused=bool(interrupts), interrupts=interrupts,
                      checkpoint_id=(snapshot.config or {}).get("configurable", {}).get("checkpoint_id"))
        result.setdefault("status", "empty")
        result["checkpoint_saved"] = bool(snapshot.values)
        if snapshot.next:
            result["state_saved"] = False
            result["turn"] = snapshot.values.get("revision", 0) + 1
        result.setdefault("turn", snapshot.values.get("revision", 0))
        result.setdefault("state_saved", False)
        result.setdefault("candidates", [])
        if "fit_review" in snapshot.next:
            result["review_pending"] = True
        if any(task.error for task in snapshot.tasks):
            result.update(status="failed", resumable=True, error="该节点上次执行失败，修复原因后可使用 --retry。")
        return result
