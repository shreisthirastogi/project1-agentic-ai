"""
dashboard.py — Streamlit Human-in-the-Loop Approval Dashboard for Project 1
Run: streamlit run frontend/dashboard.py
"""
import streamlit as st
import requests
import json

import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Ops Agent Dashboard", layout="wide")
st.title("🤖 Autonomous Ops Agent — Approval Dashboard")
st.caption("Planner → Executor → Critic → **Human Gate** → Action")

# ── Session state init ───────────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "state" not in st.session_state:
    st.session_state.state = None

# ── Sidebar: API health ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    try:
        h = requests.get(f"{API_URL}/health", timeout=2)
        if h.status_code == 200:
            st.success("✅ Backend: Connected")
        else:
            st.error("❌ Backend: Error")
    except Exception:
        st.error("❌ Backend: Not running\nStart with:\n`uvicorn main:app --reload`")

    st.divider()
    st.subheader("Run History")
    try:
        threads = requests.get(f"{API_URL}/threads", timeout=2).json()
        for tid, info in threads.items():
            status = "⏳ Awaiting Approval" if info["pending_action"] else "✅ Done"
            st.markdown(f"**{tid[:8]}…** — {status}")
    except Exception:
        st.info("No active threads.")

# ── Section 1: Submit a goal ─────────────────────────────────────────────────
st.header("1️⃣  Submit Ops Goal")
goal = st.text_area(
    "Ops Request",
    value="Check if any of our vendor invoices this month look anomalous and flag the top 3 to the #finance-alerts channel.",
    height=80,
)

if st.button("🚀 Run Agent", type="primary"):
    with st.spinner("Agent planning → executing → critic reviewing…"):
        try:
            res = requests.post(f"{API_URL}/start", json={"goal": goal}, timeout=120)
            if res.status_code == 200:
                data = res.json()
                st.session_state.thread_id = data["thread_id"]
                st.session_state.state = data["state"]
                if data["state"].get("pending_action"):
                    st.warning("⏸️ Agent paused — action requires your approval below.")
                else:
                    st.success("✅ Agent completed without requiring any actions.")
            else:
                st.error(f"Error {res.status_code}: {res.text}")
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to backend. Is `uvicorn main:app --reload` running?")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

# ── Section 2: Live Status ───────────────────────────────────────────────────
if st.session_state.thread_id:
    st.divider()
    st.header("2️⃣  Agent Status")

    # Refresh state
    try:
        r = requests.get(f"{API_URL}/status/{st.session_state.thread_id}", timeout=5)
        if r.status_code == 200:
            st.session_state.state = r.json()["state"]
    except Exception:
        pass

    state = st.session_state.state or {}

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Iterations", state.get("iterations", 0))
    col2.metric("Cost (USD)", f"${state.get('cost_usd', 0.0):.4f}")
    col3.metric("Replans", state.get("replan_count", 0))
    col4.metric("Sub-tasks", len(state.get("sub_tasks", [])))

    st.subheader("📋 Sub-tasks Planned")
    for i, t in enumerate(state.get("sub_tasks", []), 1):
        done = i - 1 < state.get("current_task_idx", 0)
        icon = "✅" if done else ("🔄" if i - 1 == state.get("current_task_idx", 0) else "⬜")
        st.markdown(f"{icon} **{i}.** {t}")

    st.subheader("🔍 Evidence Collected")
    for ev in state.get("evidence", []):
        with st.expander(f"[{ev.get('tool', 'N/A')}] — {ev.get('task', '')[:60]}…"):
            st.code(ev.get("result", ""), language=None)

    # ── Section 3: Approval Gate ─────────────────────────────────────────────
    pending = state.get("pending_action")
    if pending:
        st.divider()
        st.header("3️⃣  ⚠️ Action Requires Your Approval")
        st.error(
            f"**Tool:** `{pending.get('tool', 'unknown')}`\n\n"
            f"**Arguments:** ```{json.dumps(pending.get('args', {}), indent=2)}```"
        )
        st.info(
            "This action is **reversible** — you can rollback after approval "
            "using the Rollback button."
        )

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("✅ Approve & Execute", type="primary"):
                try:
                    ar = requests.post(
                        f"{API_URL}/approve",
                        json={"thread_id": st.session_state.thread_id, "status": "approved"},
                        timeout=15,
                    )
                    if ar.status_code == 200:
                        st.success("Action executed!")
                        st.caption(ar.json().get("rollback_hint", ""))
                        st.session_state.state = ar.json().get("new_state", state)
                        st.rerun()
                    else:
                        st.error(ar.text)
                except Exception as e:
                    st.error(str(e))

        with col_b:
            if st.button("❌ Reject Action"):
                try:
                    ar = requests.post(
                        f"{API_URL}/approve",
                        json={"thread_id": st.session_state.thread_id, "status": "rejected"},
                        timeout=15,
                    )
                    if ar.status_code == 200:
                        st.warning("Action rejected. No message sent.")
                        st.session_state.state = ar.json().get("new_state", state)
                        st.rerun()
                    else:
                        st.error(ar.text)
                except Exception as e:
                    st.error(str(e))
    else:
        st.divider()
        st.success("✅ No pending actions. Run complete.")

    # ── Section 4: Rollback ──────────────────────────────────────────────────
    st.divider()
    st.header("4️⃣  Rollback Last Action")
    st.caption("Click to undo the last executed Slack message.")
    if st.button("↩️ Rollback Last Action"):
        try:
            rb = requests.delete(
                f"{API_URL}/rollback/{st.session_state.thread_id}", timeout=10
            )
            if rb.status_code == 200:
                st.success(rb.json()["rollback_result"])
            else:
                st.error(rb.text)
        except Exception as e:
            st.error(str(e))

    # ── Section 5: Audit Trail ───────────────────────────────────────────────
    st.divider()
    with st.expander("📜 Full Audit Trail (run_log)"):
        st.json(state.get("run_log", []))
