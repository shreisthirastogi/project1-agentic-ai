"""
main.py — FastAPI backend for Autonomous Ops Agent
Handles: start run, pause on pending action, resume after approval, rollback.
"""
import uuid
import os
import json
from typing import Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import the compiled LangGraph
from agent import compiled_graph

app = FastAPI(title="Autonomous Ops Agent API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request/Response models ──────────────────────────────────────────────────
class GoalRequest(BaseModel):
    goal: str

class ApprovalRequest(BaseModel):
    thread_id: str
    status: str  # "approved" | "rejected"


# ── Helper: make state JSON-serialisable ─────────────────────────────────────
def _serialise(state: dict) -> dict:
    try:
        json.dumps(state)
        return state
    except TypeError:
        return {k: str(v) for k, v in state.items()}


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/start")
def start_agent(req: GoalRequest):
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "goal": req.goal,
        "sub_tasks": [],
        "current_task_idx": 0,
        "evidence": [],
        "pending_action": None,
        "approval_status": None,
        "iterations": 0,
        "cost_usd": 0.0,
        "replan_count": 0,
        "run_log": [],
    }

    try:
        # Run graph. It will pause if it hits the "action" node.
        for _ in compiled_graph.stream(initial_state, config=config):
            pass
        state = compiled_graph.get_state(config).values
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {e}")

    return {"thread_id": thread_id, "state": _serialise(state)}


@app.get("/status/{thread_id}")
def get_status(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    state_wrapper = compiled_graph.get_state(config)
    if not state_wrapper.values:
        raise HTTPException(status_code=404, detail="Thread not found")
    return {"state": _serialise(state_wrapper.values)}


@app.post("/approve")
def approve_action(req: ApprovalRequest):
    config = {"configurable": {"thread_id": req.thread_id}}
    state_wrapper = compiled_graph.get_state(config)
    if not state_wrapper.values:
        raise HTTPException(status_code=404, detail="Thread not found")

    state = state_wrapper.values
    if not state.get("pending_action"):
        raise HTTPException(status_code=400, detail="No pending action to approve")

    if req.status not in ("approved", "rejected"):
        raise HTTPException(status_code=400, detail="Status must be 'approved' or 'rejected'")

    # Update state with approval status
    compiled_graph.update_state(config, {"approval_status": req.status})

    # Resume the graph from the interrupted action node
    try:
        for _ in compiled_graph.stream(None, config=config):
            pass
        new_state = compiled_graph.get_state(config).values
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resuming graph: {e}")

    rollback_hint = None
    if req.status == "approved":
        staged_id = (state.get("pending_action") or {}).get("staged_id", "N/A")
        rollback_hint = f"To rollback: call DELETE /rollback/{req.thread_id} (staged_id={staged_id})"

    return {
        "status": "processed",
        "action_taken": req.status,
        "rollback_hint": rollback_hint,
        "new_state": _serialise(new_state),
    }


@app.delete("/rollback/{thread_id}")
def rollback_action(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    state_wrapper = compiled_graph.get_state(config)
    if not state_wrapper.values:
        raise HTTPException(status_code=404, detail="Thread not found")

    state = state_wrapper.values
    log = state.get("run_log", [])
    last_action = next((e for e in reversed(log) if e.get("node") == "action"), None)

    if not last_action or last_action.get("status") != "executed":
        raise HTTPException(status_code=400, detail="No executed action found to rollback")

    staged_id = (last_action.get("action") or {}).get("staged_id", "unknown")
    slack_ts = (last_action.get("action") or {}).get("slack_ts")
    channel = (last_action.get("action") or {}).get("args", {}).get("channel", "")

    rollback_result = f"ROLLBACK OK: Slack message {staged_id} deleted."
    
    # Real Slack rollback
    slack_token = os.getenv("SLACK_BOT_TOKEN")
    if slack_token and slack_ts and channel:
        try:
            from slack_sdk import WebClient
            client = WebClient(token=slack_token)
            client.chat_delete(channel=channel, ts=slack_ts)
            rollback_result += f" (Deleted from {channel})"
        except Exception as e:
            rollback_result = f"ROLLBACK FAILED: {e}"

    log.append({"node": "rollback", "staged_id": staged_id, "result": rollback_result})
    
    # Update state
    compiled_graph.update_state(config, {"run_log": log})

    return {"rollback_result": rollback_result}


@app.get("/threads")
def list_threads():
    # MemorySaver doesn't expose all thread_ids externally.
    # Return empty dict so the dashboard for-loop works without crashing.
    return {}
