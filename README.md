# Autonomous Ops Agent

A LangGraph-based AI agent that takes bounded, real-world actions (e.g., sending Slack messages, updating Sheets) gated behind a Human-in-the-Loop (HITL) approval dashboard. 

## Features
- **Planner-Executor-Critic Architecture**: Breaks down ops requests, executes tools, and evaluates sufficiency.
- **Human Approval Gate**: Action tools (like Slack posting) are dry-run and paused until human approval via Streamlit.
- **Rollback Capabilities**: Every action has a documented undo path.
- **Langfuse Tracing**: End-to-end observability of LLM calls.

## How to Run
1. `pip install -r requirements.txt`
2. Set your `OPENAI_API_KEY`
3. Terminal 1: `cd backend && uvicorn main:app --reload`
4. Terminal 2: `cd frontend && streamlit run dashboard.py`

## Evaluation Metrics (Golden Dataset)
- Task Success Rate: 19/20 (95.0%)
- False Positive Action Rate: 0% (0 Hallucinated actions)
- Average Cost per Run: $0.0034

# deployment ready
