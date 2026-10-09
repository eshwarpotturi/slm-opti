"""Prints the ladder for one model: accuracy by chore type, wrong write actions, cost.
   python3 ladder_report.py bedrock-us.meta.llama3-1-8b-instruct-v1-0 b"""
import json, os, sys
from collections import defaultdict, Counter
from score import call_ok
here = os.path.dirname(os.path.abspath(__file__))
T = {t["id"]: t for t in json.load(open(os.path.join(here, "tasks.json")))}
WRITE = {"create_invoice", "send_invoice", "send_invoice_reminder", "cancel_sent_invoice", "record_payment_for_invoice", "create_refund",
         "accept_dispute_claim", "create_shipment_tracking", "cancel_subscription", "create_product"}
cats = ["1_direct", "2_lookup", "3_list_create", "4_multi_step", "5_should_not_act"]
def wrong_write(r):
    return any(c["tool"] in WRITE and not any(call_ok(c, g) for g in T[r["id"]]["expected"]) for c in r["calls"])
model, tag = sys.argv[1], sys.argv[2]
print("%-8s %4s %6s | %s | %11s | %8s | %s" % ("layers", "n", "right", "  ".join(c[2:8].ljust(6) for c in cats), "wrong write", "cost $", "calls/chore"))
for L in ["0", "1", "12", "123", "1234", "12345"]:
    f = os.path.join(here, "results", "%s__L%s-%s.jsonl" % (model, L, tag))
    if not os.path.exists(f): continue
    rs = list({json.loads(l)["id"]: json.loads(l) for l in open(f)}.values())
    by = defaultdict(list)
    for r in rs: by[r["category"]].append(r["pass"])
    print("%-8s %4d %5.0f%% | %s | %11d | %8.4f | %.2f  esc=%d" % (L, len(rs), 100.0 * sum(r["pass"] for r in rs) / len(rs),
          "  ".join(("%3.0f%%" % (100.0 * sum(by[c]) / len(by[c])) if by[c] else " - ").ljust(6) for c in cats),
          sum(wrong_write(r) for r in rs), sum(r.get("cost", 0) for r in rs), sum(r.get("model_calls", 1) for r in rs) / len(rs), sum(r.get("escalated", False) for r in rs)))
    if len(sys.argv) > 3 and L == sys.argv[3]:
        seen = Counter()
        for r in rs:
            if r["pass"]: continue
            k = (r["category"][2:8], r["reason"]); seen[k] += 1
            if seen[k] > 2: continue
            t = T[r["id"]]
            print("   ", r["id"], k, "|", t["request"][:120]); print("       want:", json.dumps(t["expected"])[:200]); print("       got :", json.dumps(r["calls"])[:200]); print("       rej :", r.get("rejections", [])[:2])
        print("   ", dict(seen))
