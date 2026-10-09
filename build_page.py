"""Builds index.html, the results page, from the files in results/. Run: python3 build_page.py"""
import json
import os
from collections import defaultdict

from score import call_ok

HERE = os.path.dirname(os.path.abspath(__file__))
WRITE = {"create_invoice", "send_invoice", "send_invoice_reminder", "cancel_sent_invoice", "record_payment_for_invoice",
         "create_refund", "accept_dispute_claim", "create_shipment_tracking", "cancel_subscription", "create_product"}
CATS = [("1_direct", "ID given in the request"), ("2_lookup", "Find the customer's record first"),
        ("3_list_create", "Lists, dates, creating things"), ("4_multi_step", "Two or three requests in one message"),
        ("5_should_not_act", "Should stop and ask")]


def load(name):
    p = os.path.join(HERE, "results", name + ".jsonl")
    return {json.loads(l)["id"]: json.loads(l) for l in open(p)}


def wrong_write(r, t):
    return any(c["tool"] in WRITE and not any(call_ok(c, g) for g in t["expected"]) for c in r["calls"])


def show(calls):
    out = []
    for c in calls:
        args = []
        for k, v in (c.get("args") or {}).items():
            if isinstance(v, dict):
                if "$absent" in v:
                    continue
                v = v.get("$absent_or")
                if k in ("amount_value", "currency_code"):
                    continue
            if k in ("note", "note_to_payer", "reason", "item_name") or v in (None, ""):
                continue
            args.append("%s=%s" % (k, v))
        out.append("%s(%s)" % (c["tool"], ", ".join(args)))
    return out


def summary(tasks, runs):
    rows = []
    for rs in runs:
        by = defaultdict(list)
        for t in tasks:
            by[t["category"]].append(rs[t["id"]]["pass"])
        rows.append({"right": sum(rs[t["id"]]["pass"] for t in tasks), "n": len(tasks),
                     "wrong": sum(wrong_write(rs[t["id"]], t) for t in tasks),
                     "cats": [round(100.0 * sum(by[c]) / len(by[c])) for c, _ in CATS],
                     "cost": round(sum(rs[t["id"]].get("cost", 0) for t in tasks), 4),
                     "esc": sum(bool(rs[t["id"]].get("escalated")) for t in tasks)})
    return rows


G = "bedrock-google.gemma-3-4b-it"
tasks = json.load(open(os.path.join(HERE, "tasks_third.json")))
alone, four, five = load(G + "__L0-t"), load(G + "__L1234-t"), load(G + "__L12345-t")
head = summary(tasks, [alone, four, five])

dev_tasks = json.load(open(os.path.join(HERE, "tasks.json")))
dev = []
for label, stem in [("Gemma 3 4B", G), ("Llama 3.1 8B", "bedrock-us.meta.llama3-1-8b-instruct-v1-0"),
                    ("Ministral 3B", "bedrock-mistral.ministral-3-3b-instruct")]:
    pts, wr = [], []
    for L in ["0", "1", "12", "123", "1234"]:
        rs = load("%s__L%s-b" % (stem, L))
        pts.append(round(100.0 * sum(rs[t["id"]]["pass"] for t in dev_tasks) / len(dev_tasks)))
        wr.append(sum(wrong_write(rs[t["id"]], t) for t in dev_tasks))
    dev.append({"name": label, "right": pts, "wrong": wr})

rows = []
for t in tasks:
    a, b, c = alone[t["id"]], four[t["id"]], five[t["id"]]
    rows.append({"id": t["id"], "cat": [k for k, _ in CATS].index(t["category"]), "q": t["request"],
                 "want": show(t["expected"]), "why": t.get("why_no_action", ""),
                 "a": show(a["calls"]), "ap": a["pass"], "aw": wrong_write(a, t),
                 "b": show(b["calls"]), "bp": b["pass"], "rej": list(dict.fromkeys(c.get("rejections") or b.get("rejections") or []))[:2],
                 "c": show(c["calls"]), "cp": c["pass"], "esc": bool(c.get("escalated"))})

# the opening example: the model alone would have moved money wrongly, the layers stopped it
cands = [r for r in rows if r["cat"] == 4 and r["aw"] and r["cp"] and not r["c"] and r["rej"] and r["a"] and "refund" in r["a"][0]]
hero = next((r for r in cands if "REFUNDED" in r["rej"][0]), cands[0])

data = {"head": head, "cats": [n for _, n in CATS], "dev": dev, "rows": rows, "hero": hero}
html = open(os.path.join(HERE, "page_template.html")).read().replace("/*DATA*/", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
open(os.path.join(HERE, "index.html"), "w").write(html)
print("index.html written:", len(html) // 1024, "KB |", head)
print("hero:", hero["q"], "|", hero["a"], "|", hero["rej"])
