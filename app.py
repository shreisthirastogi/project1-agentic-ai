import os
import streamlit as st
import json

st.set_page_config(page_title="Ops Agent Dashboard", layout="wide")
st.title("🤖 Autonomous Ops Agent — Approval Dashboard")
st.caption("Planner → Executor → Critic → **Human Gate** → Action")

# ── Load API key from Streamlit secrets or env ────────────────────────────────
openai_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
anthropic_key = st.secrets.get("ANTHROPIC_API_KEY", os.getenv("ANTHROPIC_API_KEY", ""))
gemini_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
if gemini_key: os.environ["GEMINI_API_KEY"] = gemini_key

if not openai_key and not anthropic_key and not gemini_key:
    st.error("⚠️ No API key found. Add OPENAI_API_KEY in Streamlit Cloud secrets.")
    st.stop()

os.environ["OPENAI_API_KEY"] = openai_key
if anthropic_key:
    os.environ["ANTHROPIC_API_KEY"] = anthropic_key

# ── Import agent (now that env vars are set) ──────────────────────────────────
try:
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).parent / "backend"))
    from agent import compiled_graph
except Exception as e:
    st.error(f"Failed to load agent: {e}")
    st.stop()

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in [("thread_id", None), ("state", None), ("config", None)]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Section 1: Submit goal ────────────────────────────────────────────────────
st.header("1️⃣ Submit Ops Goal")
goal = st.text_area(
    "Ops Request",
    value="Check if any vendor invoices this month look anomalous and flag the top 3 to #finance-alerts.",
    height=80,
)

if st.button("🚀 Run Agent", type="primary"):
    import uuid
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {
        "goal": goal, "sub_tasks": [], "current_task_idx": 0,
        "evidence": [], "pending_action": None, "approval_status": None,
        "iterations": 0, "cost_usd": 0.0, "replan_count": 0, "run_log": [],
    }
    with st.spinner("Agent planning → executing → critic reviewing…"):
        try:
            for _ in compiled_graph.stream(initial_state, config=config):
                pass
            state = compiled_graph.get_state(config).values
            st.session_state.thread_id = thread_id
            st.session_state.config = config
            st.session_state.state = dict(state)
            if state.get("pending_action"):
                st.warning("⏸️ Agent paused — action requires your approval below.")
            else:
                st.success("✅ Agent completed.")
        except Exception as e:
            st.error(f"Agent error: {e}")

# ── Section 2: Live status ────────────────────────────────────────────────────
if st.session_state.state:
    st.divider()
    st.header("2️⃣ Agent Status")
    state = st.session_state.state

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
        with st.expander(f"[{ev.get('tool','N/A')}] — {str(ev.get('task',''))[:60]}…"):
            st.code(ev.get("result", ""), language=None)

    # ── Approval gate ─────────────────────────────────────────────────────────
    pending = state.get("pending_action")
    if pending:
        st.divider()
        st.header("3️⃣ ⚠️ Action Requires Your Approval")
        st.error(f"**Tool:** `{pending.get('tool')}`\n\n**Args:** ```{json.dumps(pending.get('args',{}), indent=2)}```")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("✅ Approve & Execute", type="primary"):
                config = st.session_state.config
                compiled_graph.update_state(config, {"approval_status": "approved"})
                for _ in compiled_graph.stream(None, config=config): pass
                st.session_state.state = dict(compiled_graph.get_state(config).values)
                st.success("Action executed!")
                st.rerun()
        with col_b:
            if st.button("❌ Reject Action"):
                config = st.session_state.config
                compiled_graph.update_state(config, {"approval_status": "rejected"})
                for _ in compiled_graph.stream(None, config=config): pass
                st.session_state.state = dict(compiled_graph.get_state(config).values)
                st.warning("Rejected.")
                st.rerun()
    else:
        st.divider()
        st.success("✅ No pending actions. Run complete.")

    st.divider()
    with st.expander("📜 Full Audit Trail"):
        st.json(state.get("run_log", []))