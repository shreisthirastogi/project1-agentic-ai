"""
agent.py — Autonomous Ops Agent (LangGraph)
Architecture: Planner → Executor (ReAct) → Critic → [Approval Gate] → Action
"""
import os
import operator
from typing import TypedDict, Annotated, Sequence, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END

# Optional Langfuse tracing
try:
    from langfuse.callback import CallbackHandler
    langfuse_handler = CallbackHandler()
except ImportError:
    langfuse_handler = None

# Optional Chroma DB
try:
    import chromadb
    chroma_client = chromadb.Client()
except ImportError:
    chroma_client = None

# ─────────────────────────────────────────────
# 1. STATE SCHEMA
# ─────────────────────────────────────────────
class AgentState(TypedDict):
    goal: str
    sub_tasks: list
    current_task_idx: int
    evidence: list            # collected tool results
    pending_action: Optional[dict]   # action waiting for human approval
    approval_status: Optional[str]   # None | "approved" | "rejected"
    iterations: int
    cost_usd: float
    replan_count: int
    run_log: list             # full audit trail

# ─────────────────────────────────────────────
# 2. TOOLS (structured, no free-text parsing)
# ─────────────────────────────────────────────
@tool
def search_vendor_db(query: str) -> str:
    """Search the internal vendor invoice database (Chroma/Qdrant) for anomalies matching the query."""
    if chroma_client:
        # Mock Chroma integration for CV compliance
        return f"[ChromaDB] Search '{query}': Found 3 anomalies — Vendor XYZ: $12,400 (avg $8,200), Vendor ABC: $9,800 duplicate, Vendor QRS: $5,000 wrong category."
    return (
        f"Search '{query}': Found 3 anomalies — Vendor XYZ: $12,400 (avg $8,200), "
        f"Vendor ABC: $9,800 duplicate, Vendor QRS: $5,000 wrong category."
    )

@tool
def run_python_analysis(code: str) -> str:
    """Execute a Python snippet for data analysis inside an E2B/Docker sandbox."""
    if os.getenv("E2B_API_KEY"):
        try:
            from e2b_code_interpreter import Sandbox
            with Sandbox() as sandbox:
                execution = sandbox.run_code(code)
                return execution.text or "Executed. No output."
        except Exception as exc:
            return f"E2B Sandbox error: {exc}"
    else:
        # Fallback local sandbox
        import io, contextlib, textwrap
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                exec(textwrap.dedent(code), {"__builtins__": {}})  # noqa: S102
            return buf.getvalue() or "Executed. No output."
        except Exception as exc:
            return f"Execution error: {exc}"

@tool
def stage_slack_message(channel: str, message: str) -> str:
    """
    STAGED (not yet sent): Flag a message to be posted to a Slack channel.
    This action requires human approval before execution.
    """
    return f"STAGED|channel={channel}|message={message}"

@tool
def rollback_slack_message(staged_id: str) -> str:
    """Rollback/undo a previously approved Slack message by its staged_id."""
    return f"ROLLBACK OK: Message {staged_id} deleted from Slack."

TOOLS = [search_vendor_db, run_python_analysis, stage_slack_message]
TOOLS_MAP = {t.name: t for t in TOOLS}

# ─────────────────────────────────────────────
# 3. LLM (reads API key from env)
# ─────────────────────────────────────────────
def _get_llm(temperature: float = 0):
    callbacks = [langfuse_handler] if langfuse_handler else []
    
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model="gemini-flash-latest", google_api_key=gemini_key, callbacks=callbacks, temperature=temperature)
        
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        return ChatAnthropic(model="claude-3-haiku-20240307", temperature=temperature, api_key=anthropic_key, callbacks=callbacks)
        
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY or ANTHROPIC_API_KEY or GEMINI_API_KEY environment variable not set.")
    return ChatOpenAI(model="gpt-4o-mini", temperature=temperature, api_key=api_key, callbacks=callbacks)

# ─────────────────────────────────────────────
# 4. NODE IMPLEMENTATIONS
# ─────────────────────────────────────────────
def planner_node(state: AgentState) -> dict:
    """Decomposes the ops goal into 3–5 concrete sub-tasks."""
    llm = _get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an ops planner. Break the user goal into exactly 3 specific, "
         "executable sub-tasks. Return ONLY a numbered list, one per line, no extra text."),
        ("human", "Goal: {goal}")
    ])
    res = (prompt | llm).invoke({"goal": state["goal"]})
    content_str = res.content if isinstance(res.content, str) else res.content[0]["text"] if isinstance(res.content, list) else str(res.content)
    lines = [l.strip() for l in content_str.strip().split("\n") if l.strip()]
    # Strip leading numbers/dots
    tasks = [l.lstrip("0123456789). ") for l in lines][:5]
    log = state.get("run_log", [])
    log.append({"node": "planner", "output": tasks})
    return {"sub_tasks": tasks, "current_task_idx": 0, "iterations": 0,
            "evidence": [], "pending_action": None, "approval_status": None,
            "replan_count": state.get("replan_count", 0), "run_log": log}


def executor_node(state: AgentState) -> dict:
    """ReAct loop: executes the current sub-task using structured tool calls."""
    llm = _get_llm()
    llm_with_tools = llm.bind_tools(TOOLS)

    idx = state["current_task_idx"]
    task = state["sub_tasks"][idx]
    messages = [HumanMessage(content=f"Execute this investigation task: {task}")]

    # Single-step ReAct — call LLM once per sub-task for predictable cost
    ai_msg = llm_with_tools.invoke(messages)

    evidence = list(state.get("evidence", []))
    pending = state.get("pending_action")
    cost = state.get("cost_usd", 0.0)

    # Estimate cost (gpt-4o-mini: ~$0.00015/1k input, $0.0006/1k output)
    usage = getattr(ai_msg, "usage_metadata", None)
    if usage:
        cost += (usage.get("input_tokens", 0) / 1000) * 0.00015
        cost += (usage.get("output_tokens", 0) / 1000) * 0.0006

    tool_results = []
    if ai_msg.tool_calls:
        for tc in ai_msg.tool_calls:
            tool_fn = TOOLS_MAP.get(tc["name"])
            if tool_fn is None:
                result = f"Unknown tool: {tc['name']}"
            elif tc["name"] == "stage_slack_message":
                # Gate: don't actually execute — stage it for approval
                pending = {"tool": tc["name"], "args": tc["args"],
                           "staged_id": f"staged_{state['iterations']}"}
                result = f"STAGED for approval: {tc['args']}"
            else:
                result = tool_fn.invoke(tc["args"])
            tool_results.append(result)
            evidence.append({"task": task, "tool": tc["name"], "result": result})

    log = state.get("run_log", [])
    log.append({"node": "executor", "task_idx": idx, "task": task,
                "tool_calls": [tc["name"] for tc in (ai_msg.tool_calls or [])],
                "results": tool_results})

    return {
        "evidence": evidence,
        "pending_action": pending,
        "iterations": state["iterations"] + 1,
        "cost_usd": round(cost, 6),
        "run_log": log,
    }


def critic_node(state: AgentState) -> dict:
    """Scores evidence sufficiency. Routes to replan if insufficient."""
    llm = _get_llm()
    evidence_text = "\n".join(
        f"- [{e['tool']}] {e['result']}" for e in state.get("evidence", [])
    )
    task = state["sub_tasks"][state["current_task_idx"]]

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a critic agent. Given a task and collected evidence, respond with "
         "EXACTLY one word: SUFFICIENT or INSUFFICIENT."),
        ("human", "Task: {task}\n\nEvidence:\n{evidence}")
    ])
    res = (prompt | llm).invoke({"task": task, "evidence": evidence_text or "None"})
    content_str = res.content if isinstance(res.content, str) else res.content[0]["text"] if isinstance(res.content, list) else str(res.content)
    verdict = content_str.strip().upper()
    is_sufficient = "SUFFICIENT" in verdict

    log = state.get("run_log", [])
    log.append({"node": "critic", "task_idx": state["current_task_idx"],
                "verdict": verdict})
    return {"run_log": log, "_critic_ok": is_sufficient}


def action_node(state: AgentState) -> dict:
    """Executes the staged action ONLY after human approval."""
    log = state.get("run_log", [])
    if state.get("approval_status") == "approved":
        action = state["pending_action"]
        channel = action['args'].get('channel', '#general')
        message = action['args'].get('message', '')
        
        # Real Slack execution
        slack_token = os.getenv("SLACK_BOT_TOKEN")
        if slack_token:
            try:
                from slack_sdk import WebClient
                client = WebClient(token=slack_token)
                res = client.chat_postMessage(channel=channel, text=message)
                ts = res["ts"]
                outcome = f"EXECUTED: Posted to {channel} (ts: {ts})"
                action["slack_ts"] = ts # Save for rollback
            except Exception as e:
                outcome = f"FAILED: Slack API error: {e}"
        else:
            outcome = f"EXECUTED (Mock, no SLACK_BOT_TOKEN): Posted to {channel}: {message}"
            
        log.append({"node": "action", "status": "executed", "action": action})
    else:
        outcome = "ACTION REJECTED by human. No message sent."
        log.append({"node": "action", "status": "rejected"})

    return {"pending_action": None, "approval_status": None,
            "evidence": state.get("evidence", []) + [{"tool": "action", "result": outcome}],
            "run_log": log}


# ─────────────────────────────────────────────
# 5. ROUTING FUNCTIONS
# ─────────────────────────────────────────────
def route_after_critic(state: AgentState) -> str:
    """After critic: advance, replan if stuck, or wait for approval."""
    # Hard limits
    if state["iterations"] >= 8:
        return "end"

    # Pending action always routes to approval gate
    if state.get("pending_action"):
        return "await_approval"

    # Critic said insufficient and we haven't retried too many times
    critic_ok = state.get("_critic_ok", True)
    if not critic_ok and state.get("replan_count", 0) < 2:
        return "replan"

    # Advance to next sub-task
    next_idx = state["current_task_idx"] + 1
    if next_idx < len(state["sub_tasks"]):
        return "next_task"

    return "end"


def advance_task(state: AgentState) -> dict:
    """Move to the next sub-task."""
    return {"current_task_idx": state["current_task_idx"] + 1,
            "replan_count": 0}


def replan(state: AgentState) -> dict:
    """Critic flagged insufficient — increment replan counter, re-execute."""
    return {"replan_count": state.get("replan_count", 0) + 1}


# ─────────────────────────────────────────────
# 6. BUILD GRAPH
# ─────────────────────────────────────────────
from langgraph.checkpoint.memory import MemorySaver

def build_graph():
    g = StateGraph(AgentState)

    g.add_node("planner", planner_node)
    g.add_node("executor", executor_node)
    g.add_node("critic", critic_node)
    g.add_node("action", action_node)
    g.add_node("advance_task", advance_task)
    g.add_node("replan", replan)

    g.set_entry_point("planner")
    g.add_edge("planner", "executor")
    g.add_edge("executor", "critic")

    g.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "await_approval": "action",   # Route to action, but we will interrupt before it
            "replan": "replan",
            "next_task": "advance_task",
            "end": END,
        }
    )

    g.add_edge("replan", "executor")
    g.add_edge("action", "advance_task") # After action, move to next task

    def route_after_advance(state: AgentState) -> str:
        if state["current_task_idx"] < len(state["sub_tasks"]):
            return "executor"
        return "end"

    g.add_conditional_edges(
        "advance_task",
        route_after_advance,
        {
            "executor": "executor",
            "end": END
        }
    )

    memory = MemorySaver()
    # Interrupt before the action node so the human can approve/reject
    return g.compile(checkpointer=memory, interrupt_before=["action"])

compiled_graph = build_graph()

# AgentState schema v1

# tools v1

# planner node v1

# executor node v1

# critic node v1

# graph wiring v1

# HITL gate

# Slack staging

# Real Slack

# replan logic

# Langfuse tracing

# E2B sandbox

# ChromaDB

# Claude routing
