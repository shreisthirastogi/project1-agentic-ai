"""
eval_runner.py — Golden-set evaluation harness for Project 1
Run: python eval/eval_runner.py
Requires the backend to be running on localhost:8000
"""
import json
import time
import requests
from tabulate import tabulate

API_URL = "http://localhost:8000"

# ── 20-case golden eval set ──────────────────────────────────────────────────
# Each case: {goal, expected_action: bool, expected_keywords_in_evidence: list}
GOLDEN_SET = [
    {"id": 1, "goal": "Check vendor invoices for anomalies and flag top 3 to #finance", "expected_action": True, "keywords": ["anomaly", "vendor"]},
    {"id": 2, "goal": "Summarise this week's support ticket trends", "expected_action": False, "keywords": ["trend", "ticket"]},
    {"id": 3, "goal": "Find duplicate invoices from last month and alert #finance", "expected_action": True, "keywords": ["duplicate"]},
    {"id": 4, "goal": "Check if server costs exceeded budget this quarter", "expected_action": False, "keywords": ["cost", "budget"]},
    {"id": 5, "goal": "Flag any vendor with invoices 50% above their average to #finance", "expected_action": True, "keywords": ["above", "average"]},
    {"id": 6, "goal": "Analyse trip completion rates by driver category", "expected_action": False, "keywords": ["trip", "driver"]},
    {"id": 7, "goal": "Identify top 3 anomalous payments and notify #alerts channel", "expected_action": True, "keywords": ["anomal"]},
    {"id": 8, "goal": "List all vendors with missing invoice dates", "expected_action": False, "keywords": ["missing"]},
    {"id": 9, "goal": "Summarise this month's procurement spend by category", "expected_action": False, "keywords": ["spend", "category"]},
    {"id": 10, "goal": "Send a weekly ops summary to #ops-team", "expected_action": True, "keywords": ["summary"]},
    {"id": 11, "goal": "Find invoices with mismatched PO numbers", "expected_action": False, "keywords": ["mismatch", "PO"]},
    {"id": 12, "goal": "Alert finance team about invoices pending over 30 days", "expected_action": True, "keywords": ["pending", "30"]},
    {"id": 13, "goal": "Generate a report of top 5 suppliers by spend", "expected_action": False, "keywords": ["supplier", "spend"]},
    {"id": 14, "goal": "Check if any SLAs were breached this week and notify #ops", "expected_action": True, "keywords": ["SLA", "breach"]},
    {"id": 15, "goal": "Identify vendors with no activity in the last 90 days", "expected_action": False, "keywords": ["inactiv", "90"]},
    {"id": 16, "goal": "Flag unusually large single-item purchases to #finance", "expected_action": True, "keywords": ["large", "purchase"]},
    {"id": 17, "goal": "Summarise pending approvals in the procurement system", "expected_action": False, "keywords": ["pending", "approval"]},
    {"id": 18, "goal": "Send daily cost report to #finance-alerts channel", "expected_action": True, "keywords": ["cost", "report"]},
    {"id": 19, "goal": "Find all invoices marked as urgent and not yet processed", "expected_action": False, "keywords": ["urgent", "process"]},
    {"id": 20, "goal": "Notify #finance about any vendor whose total this month exceeds $50k", "expected_action": True, "keywords": ["exceed", "50k"]},
]

def run_eval():
    results = []
    true_positives = 0   # expected action=True, agent acted
    false_positives = 0  # expected action=False, agent acted (BAD)
    false_negatives = 0  # expected action=True, agent didn't act
    true_negatives = 0   # expected action=False, agent didn't act

    for case in GOLDEN_SET:
        try:
            res = requests.post(f"{API_URL}/start", json={"goal": case["goal"]}, timeout=60)
            if res.status_code != 200:
                results.append({**case, "agent_acted": "ERROR", "task_success": False, "cost": 0})
                continue

            state = res.json()["state"]
            agent_acted = bool(state.get("pending_action"))
            cost = state.get("cost_usd", 0.0)

            # Evidence keyword check
            evidence_text = " ".join(
                str(e.get("result", "")) for e in state.get("evidence", [])
            ).lower()
            kw_hit = any(kw.lower() in evidence_text for kw in case["keywords"])

            task_success = kw_hit  # Evidence contains expected signal

            # Confusion matrix tracking
            if case["expected_action"] and agent_acted:
                true_positives += 1
            elif not case["expected_action"] and agent_acted:
                false_positives += 1
            elif case["expected_action"] and not agent_acted:
                false_negatives += 1
            else:
                true_negatives += 1

            results.append({
                "id": case["id"],
                "goal": case["goal"][:50] + "…",
                "expected_action": case["expected_action"],
                "agent_acted": agent_acted,
                "kw_hit": kw_hit,
                "task_success": task_success,
                "cost_usd": cost,
            })
            time.sleep(0.5)  # Rate limiting

        except Exception as e:
            results.append({**case, "agent_acted": "ERROR", "task_success": False, "cost": 0, "error": str(e)})

    # ── Print Results ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("GOLDEN SET EVALUATION RESULTS")
    print("=" * 70)
    headers = ["ID", "Goal", "Exp.Act", "Acted", "KW✓", "Success", "Cost $"]
    rows = [
        [r["id"], r.get("goal", "")[:45], r["expected_action"],
         r["agent_acted"], r.get("kw_hit", "?"), r["task_success"],
         f"${r.get('cost_usd', 0):.4f}"]
        for r in results
    ]
    print(tabulate(rows, headers=headers, tablefmt="grid"))

    total = len(results)
    successes = sum(1 for r in results if r["task_success"])
    total_cost = sum(r.get("cost_usd", 0) for r in results)
    avg_cost = total_cost / total if total else 0

    print(f"\n📊 SUMMARY")
    print(f"  Task Success Rate    : {successes}/{total} ({successes/total*100:.1f}%)")
    print(f"  True Positives       : {true_positives}")
    print(f"  False Positives (⚠️) : {false_positives}  ← This must be 0")
    print(f"  False Negatives      : {false_negatives}")
    print(f"  True Negatives       : {true_negatives}")
    print(f"  Avg Cost / Run       : ${avg_cost:.4f}")
    print(f"  Total Cost           : ${total_cost:.4f}")

    # Save results
    with open("eval/eval_results.json", "w") as f:
        json.dump({"summary": {
            "task_success": f"{successes}/{total}",
            "false_positives": false_positives,
            "avg_cost_usd": round(avg_cost, 5),
        }, "results": results}, f, indent=2)
    print("\n✅ Results saved to eval/eval_results.json")

if __name__ == "__main__":
    run_eval()
