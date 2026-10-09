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
alone, four, five = load(G + "__L0-t"), load(G + "__L1234-u"), load(G + "__L12345-u")
BIG = "bedrock-us.anthropic.claude-sonnet-4-5-20250929-v1-0"
big0, big4 = load(BIG + "__L0-u"), load(BIG + "__L1234-u")
head = summary(tasks, [alone, four, five])
ref = summary(tasks, [big0, big4])

# every layer on the unseen test, three small models
lad = []
for label, stem in [("Gemma 3 4B", G), ("Llama 3.1 8B", "bedrock-us.meta.llama3-1-8b-instruct-v1-0"),
                    ("Ministral 3B", "bedrock-mistral.ministral-3-3b-instruct")]:
    pts, wr = [], []
    for L in ["0", "1", "12", "123", "1234"]:
        rs = alone if (stem == G and L == "0") else load("%s__L%s-u" % (stem, L))
        pts.append(round(100.0 * sum(rs[t["id"]]["pass"] for t in tasks) / len(tasks)))
        wr.append(sum(wrong_write(rs[t["id"]], t) for t in tasks))
    lad.append({"name": label, "right": pts, "wrong": wr})

rows = []
for t in tasks:
    a, b, c = alone[t["id"]], four[t["id"]], five[t["id"]]
    rows.append({"id": t["id"], "cat": [k for k, _ in CATS].index(t["category"]), "q": t["request"],
                 "want": show(t["expected"]), "why": t.get("why_no_action", ""),
                 "a": show(a["calls"]), "ap": a["pass"], "aw": wrong_write(a, t),
                 "b": show(b["calls"]), "bp": b["pass"], "rej": list(dict.fromkeys(c.get("rejections") or b.get("rejections") or []))[:2],
                 "c": show(c["calls"]), "cp": c["pass"], "esc": bool(c.get("escalated"))})



def steps(tid):
    """What each layer did for one request, from the recorded run."""
    out = []
    for tr in four[tid]["trace"]:
        kept = sum(v[0] for v in tr["records"].values()); total = sum(v[1] for v in tr["records"].values())
        out.append({"seg": tr["seg"], "kept": kept, "total": total,
                    "tools": tr["tools"] if isinstance(tr["tools"], list) else [],
                    "first": show(tr["first"]),
                    "rej": [[(show([c])[0] if c else ""), w] for c, w in tr["rejected"]],
                    "second": None if tr["second"] is None else show(tr["second"]),
                    "rej2": [[show([c])[0], w] for c, w in tr.get("rejected2", [])],
                    "final": show(tr["final"])})
    return out


used, walks = set(), []


def pick(label, test):
    r = next((r for r in rows if r["id"] not in used and test(r)), None)
    if r:
        used.add(r["id"])
        walks.append({"label": label, "id": r["id"], "steps": steps(r["id"]), "agree": five[r["id"]].get("agree")})


def tr_of(r):
    return four[r["id"]]["trace"]


pick("A refund on an order already refunded", lambda r: r["cat"] == 4 and r["aw"] and r["cp"] and any("REFUNDED" in w for t in tr_of(r) for _, w in t["rejected"]))
pick("Two requests in one message", lambda r: r["cat"] == 3 and not r["ap"] and r["bp"] and len(tr_of(r)) == 2)
pick("A wrong first answer, corrected", lambda r: r["bp"] and r["b"] and any(t["rejected"] and t["final"] for t in tr_of(r)))
pick("A customer who is not on the account", lambda r: r["why"].startswith("customer is not") and r["aw"] and r["bp"])
pick("An invented detail", lambda r: r["cat"] == 4 and r["bp"] and any("did not" in w for t in tr_of(r) for _, w in t["rejected"]))
pick("Handed to the large model", lambda r: r["esc"] and r["cp"] and not r["bp"])
pick("One the layers still get wrong", lambda r: not r["cp"] and not r["esc"] and r["b"])

# the opening example: the model alone would have moved money wrongly, the layers stopped it
cands = [r for r in rows if r["cat"] == 4 and r["aw"] and r["cp"] and not r["c"] and r["rej"] and r["a"] and "refund" in r["a"][0]]
hero = next((r for r in cands if "REFUNDED" in r["rej"][0]), cands[0])

data = {"ref": ref, "head": head, "cats": [n for _, n in CATS], "lad": lad, "rows": rows, "hero": hero, "walks": walks}
html = open(os.path.join(HERE, "page_template.html")).read().replace("/*DATA*/", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
open(os.path.join(HERE, "tools.html"), "w").write(html)
print("tools.html written:", len(html) // 1024, "KB |", head)
print("walkthroughs:", [(w["label"], w["id"], len(w["steps"])) for w in walks])
print("ladder:", lad)
print("hero:", hero["q"], "|", hero["a"], "|", hero["rej"])
