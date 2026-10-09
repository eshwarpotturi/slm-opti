"""A classical baseline with no language model.

  1. Split the message into requests            (same plain-code splitter as layer 3)
  2. Pick the action for each request           (TF-IDF features + logistic regression, trained on train_tasks.json)
  3. Fill in the details                        (hand-written rules and patterns, one block per action)
  4. Optionally, check the action against the account rules  (the same rule check as layer 4)

   python3 classic.py            trains, checks itself on held-back training requests, then runs set C once

Results go to results/classic__Lc-c.jsonl (steps 1 to 3) and results/classic__Lc4-c.jsonl (with the rule check).
Everything it prints is also written to results/classic_log.txt.
"""
import datetime as dt
import json
import os
import random
import re
import sys
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, make_union

from score import grade
from stack import COLLECTIONS, HERE, ground, guard, segments, sentences

LOG = []


def say(*a):
    line = " ".join(str(x) for x in a)
    print(line); LOG.append(line)


# ---------------------------------------------------------------- step 2: which action
TRAP_INTENT = {"order is already refunded": "create_refund", "invoice is already paid": "send_invoice_reminder",
               "subscription is already cancelled": "cancel_subscription",
               "dispute is not awaiting the merchant's action": "accept_dispute_claim",
               "customer has two orders and the request does not say which": "create_refund",
               "no tool can do this": "none"}


def normalise(text, account):
    """Names, IDs and numbers carry no signal about which action is wanted, so they are masked."""
    t = text
    for n in sorted({r["customer"] for c in COLLECTIONS for r in account[c]}, key=len, reverse=True):
        t = re.sub(re.escape(n), " NAME ", t, flags=re.I)
    kinds = {"order_id": "ORDERID", "capture_id": "ORDERID", "invoice_id": "INVOICEID", "dispute_id": "DISPUTEID",
             "subscription_id": "SUBSCRIPTIONID", "product_id": "PRODUCTID"}
    for c in COLLECTIONS + ("products",):          # an ID that is on the account is replaced by its kind
        for r in account[c]:
            for k, v in r.items():
                if k in kinds and str(v) in t:
                    t = t.replace(str(v), " %s " % kinds[k])
    t = re.sub(r"\b(?=[A-Za-z0-9-]*\d)[A-Za-z0-9-]{8,}\b", " IDCODE ", t)
    t = re.sub(r"\$\s?\d+(?:\.\d+)?", " MONEY ", t)
    t = re.sub(r"\d+(?:\.\d+)?", " NUM ", t)
    return t.lower()


def labelled(tasks):
    """(text, action) pairs. A message with several requests is used only when it splits cleanly."""
    out = []
    for t in tasks:
        ans = t.get("answer", t.get("expected"))
        if not ans:
            tool = TRAP_INTENT.get(t.get("why_no_action", ""))
            if tool:
                out.append((normalise(t["request"], t["account"]), tool))
            continue
        segs = segments(t["request"], t["account"])
        if len(ans) == 1:
            out.append((normalise(t["request"], t["account"]), ans[0]["tool"]))
        elif len(segs) == len(ans):
            out += [(normalise(s, t["account"]), c["tool"]) for s, c in zip(segs, ans)]
    return out


def train(pairs):
    clf = make_pipeline(make_union(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
                                   TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)),
                        LogisticRegression(C=20, max_iter=3000))
    clf.fit([p[0] for p in pairs], [p[1] for p in pairs])
    return clf


# ---------------------------------------------------------------- step 3: the details
MONTHS = {"september": 9, "october": 10, "sept": 9, "sep": 9, "oct": 10}


def dates(low, today):
    y, m, d = map(int, today.split("-"))
    td = dt.date(y, m, d)
    iso = lambda mo, day: dt.date(y, mo, day).isoformat()
    mon = "(" + "|".join(MONTHS) + ")"
    g = re.search(r"(\d{1,2})(?:st|nd|rd|th)?\s+%s\w*\s*(?:to|and|through|thru|until|till|-|–)\s*(\d{1,2})(?:st|nd|rd|th)?\s+%s" % (mon, mon), low)
    if g:
        return iso(MONTHS[g.group(2)], int(g.group(1))), iso(MONTHS[g.group(4)], int(g.group(3)))
    g = re.search(r"%s\w*\s+(\d{1,2})(?:st|nd|rd|th)?\s*(?:to|and|through|thru|until|till|-|–)\s*(?:%s\w*\s+)?(\d{1,2})" % (mon, mon), low)
    if g:
        return iso(MONTHS[g.group(1)], int(g.group(2))), iso(MONTHS[g.group(3) or g.group(1)], int(g.group(4)))
    g = re.search(r"(\d{1,2})(?:st|nd|rd|th)?\s*(?:to|and|through|-|–)\s*(\d{1,2})(?:st|nd|rd|th)?\s+%s" % mon, low)
    if g:
        return iso(MONTHS[g.group(3)], int(g.group(1))), iso(MONTHS[g.group(3)], int(g.group(2)))
    if "yesterday" in low:
        x = (td - dt.timedelta(days=1)).isoformat(); return x, x
    if re.search(r"\btoday\b", low):
        return today, today
    g = re.search(r"(\d{1,2})(?:st|nd|rd|th)?\s+%s" % mon, low) or None
    if g:
        x = iso(MONTHS[g.group(2)], int(g.group(1))); return x, x
    g = re.search(r"%s\w*\s+(\d{1,2})(?:st|nd|rd|th)?\b" % mon, low)
    if g:
        x = iso(MONTHS[g.group(1)], int(g.group(2))); return x, x
    g = re.search(r"\b%s" % mon, low)
    if g:
        mo = MONTHS[g.group(1)]
        last = (dt.date(y, mo % 12 + 1, 1) - dt.timedelta(days=1)).day
        return iso(mo, 1), iso(mo, last)
    return None


def choose(records, text, allowed=None, id_key=None):
    """One record for this request: the one whose ID is written out, else one in the right state, else any."""
    low = text.lower()
    for r in records:
        if id_key and str(r[id_key]).lower() in low:
            return r
    pool = [r for r in records if allowed is None or r.get("status", r.get("state")) in allowed] or list(records)
    for r in pool:
        if str(r.get("item", r.get("plan", ""))).lower() in low and r.get("item", r.get("plan")):
            return r
    return pool[0] if pool else None


def fill(tool, seg, g):
    """Returns the call for this request, or None when a needed detail cannot be found."""
    low = seg.lower()
    money = [float(x) for x in re.findall(r"\$\s?(\d+(?:\.\d+)?)", seg)]
    call = lambda **a: {"tool": tool, "args": a}
    if tool == "none":
        return None
    if tool in ("list_products", "list_subscription_plans"):
        return call()
    if tool == "create_refund":
        o = choose(g["orders"], seg, {"COMPLETED"}, "order_id")
        if not o:
            return None
        if money and money[0] < o["amount"] and "full" not in low:
            return call(capture_id=o["capture_id"], amount_value=money[0], currency_code="USD")
        if re.search(r"\bpart(ial|ly)?\b", low) and not money:
            return None                              # a partial refund with no amount stated
        return call(capture_id=o["capture_id"])
    if tool in ("get_order", "get_shipment_tracking"):
        o = choose(g["orders"], seg, None, "order_id")
        return call(order_id=o["order_id"]) if o else None
    if tool == "create_shipment_tracking":
        o = choose(g["orders"], seg, None, "order_id")
        tn = re.search(r"\b[A-Za-z]{2}\d{8,}\b", seg)
        car = next((c for c, k in (("UPS", "ups"), ("FEDEX", "fedex"), ("DHL", "dhl"), ("USPS", "usps")) if re.search(r"\b%s\b" % k, low)), None)
        return call(order_id=o["order_id"], tracking_number=tn.group(0).upper(), carrier=car) if (o and tn and car) else None
    if tool in ("get_invoice", "send_invoice", "send_invoice_reminder", "cancel_sent_invoice", "generate_invoice_qr_code", "record_payment_for_invoice"):
        want = {"send_invoice": {"DRAFT"}, "get_invoice": None, "generate_invoice_qr_code": {"SENT"}}.get(tool, {"SENT"})
        v = choose(g["invoices"], seg, want, "invoice_id")
        if not v:
            return None
        if tool == "record_payment_for_invoice":
            method = "CASH" if "cash" in low else "CHECK" if re.search(r"\bche(ck|que)\b", low) else \
                     "BANK_TRANSFER" if re.search(r"bank|transfer|wire", low) else "OTHER"
            return call(invoice_id=v["invoice_id"], amount_value=money[0] if money else v["amount"], currency_code="USD", method=method)
        return call(invoice_id=v["invoice_id"])
    if tool == "create_invoice":
        who = next((r for c in ("orders", "invoices") for r in g[c]), None)
        rest = re.sub(r"\$\s?\d+(?:\.\d+)?", " ", seg)
        qty = re.search(r"\b(\d{1,3})\s*x\b", rest, re.I) or re.search(r"(?<![\d.])\b(\d{1,3})\b(?!\.\d)", rest)
        if not (who and money and qty):
            return None
        return call(recipient_email=who["email"], item_name="item", quantity=int(qty.group(1)), unit_amount=money[0], currency_code="USD")
    if tool in ("get_dispute", "accept_dispute_claim"):
        d = choose(g["disputes"], seg, {"REQUIRED_ACTION"} if tool == "accept_dispute_claim" else None, "dispute_id")
        if not d:
            return None
        return call(dispute_id=d["dispute_id"], note="Claim accepted") if tool == "accept_dispute_claim" else call(dispute_id=d["dispute_id"])
    if tool == "list_disputes":
        st = "UNDER_REVIEW" if "review" in low else "RESOLVED" if re.search(r"resolv|sorted|closed|settled", low) else \
             "REQUIRED_ACTION" if re.search(r"action|need|waiting on me|respond", low) else None
        return call(dispute_state=st) if st else call()
    if tool in ("show_subscription_details", "cancel_subscription"):
        s = choose(g["subscriptions"], seg, {"ACTIVE"} if tool == "cancel_subscription" else None, "subscription_id")
        if not s:
            return None
        return call(subscription_id=s["subscription_id"], reason="Customer asked to cancel") if tool == "cancel_subscription" else call(subscription_id=s["subscription_id"])
    if tool == "show_product_details":
        p = choose(g["products"], seg, None, "product_id")
        return call(product_id=p["product_id"]) if p else None
    if tool == "create_product":
        name = re.search(r'"([^"]+)"', seg) or re.search(r'"([^"]{2,})$', seg.strip())
        typ = "PHYSICAL" if "physical" in low else "DIGITAL" if "digital" in low else "SERVICE" if "service" in low else None
        return call(name=name.group(1), type=typ) if (name and typ) else None
    if tool == "list_invoices":
        st = "DRAFT" if "draft" in low else "CANCELLED" if "cancel" in low else \
             "SENT" if re.search(r"unpaid|not paid|sent|outstanding|still owe|haven'?t paid", low) else "PAID" if "paid" in low else None
        return call(status=st) if st else call()
    if tool == "list_transactions":
        d = dates(low, g["today"])
        return call(start_date=d[0], end_date=d[1]) if d else None
    if tool == "get_refund":
        known = {str(v) for c in COLLECTIONS for r in g[c] for k, v in r.items() if k.endswith("_id")}
        tok = [x for x in re.findall(r"\b(?=[A-Z0-9]*\d)[A-Z0-9]{10,}\b", seg) if x not in known]
        return call(refund_id=tok[0]) if tok else None
    return None


# ---------------------------------------------------------------- the pipeline
SENTENCE_CONFIDENCE = 0.6
def solve(clf, task, use_rules):
    calls, trace = [], []
    for seg in segments(task["request"], task["account"]):
        # one request can still hold two actions ("refund her, and list my plans"), so each sentence is classified too
        units = [seg] + (sentences(seg) if len(sentences(seg)) > 1 else [])
        picked = {}
        for i, u in enumerate(units):
            p = clf.predict_proba([normalise(u, task["account"])])[0]
            tool, conf = str(clf.classes_[p.argmax()]), float(p.max())
            if (i == 0 or conf >= SENTENCE_CONFIDENCE) and tool not in picked:
                picked[tool] = conf
        g = ground(task["account"], seg)
        for tool, conf in picked.items():
            c = fill(tool, seg, g)
            why = guard(c, task["account"], task["request"]) if (c and use_rules) else None
            trace.append({"seg": seg, "picked": tool, "confidence": round(conf, 3), "filled": c is not None, "rule": why})
            if c and not why and c not in calls:
                calls.append(c)
    return calls, trace


def run(clf, tasks, use_rules, out=None):
    rs = []
    for t in tasks:
        calls, trace = solve(clf, t, use_rules)
        ok, why = grade(calls, t["expected"])
        rs.append({"id": t["id"], "category": t["category"], "pass": ok, "reason": why, "calls": calls,
                   "escalated": False, "cost": 0.0, "model_calls": 0, "trace": trace})
    if out:
        with open(os.path.join(HERE, "results", out), "w") as f:
            for r in rs:
                f.write(json.dumps(r) + "\n")
    return rs


def report(label, rs):
    by = {}
    for r in rs:
        by.setdefault(r["category"][2:], []).append(r["pass"])
    say("%-34s right %3d of %d (%.0f%%)  |  %s" % (label, sum(r["pass"] for r in rs), len(rs), 100.0 * sum(r["pass"] for r in rs) / len(rs),
                                                    "  ".join("%s %.0f%%" % (k, 100.0 * sum(v) / len(v)) for k, v in sorted(by.items()))))


if __name__ == "__main__":
    t0 = time.time()
    data = json.load(open(os.path.join(HERE, "train_tasks.json")))
    random.Random(7).shuffle(data)
    cut = int(len(data) * 0.7)
    say("Classical baseline: TF-IDF + logistic regression for the action, hand-written rules for the details.")
    say("Training requests: %d. Held back for self-check: %d." % (cut, len(data) - cut))

    # self-check on held-back training requests (this is what the rules were developed against)
    clf = train(labelled(data[:cut]))
    held = labelled(data[cut:])
    say("Action picked correctly on held-back requests: %.1f%% of %d" % (100.0 * sum(clf.predict([x])[0] == y for x, y in held) / len(held), len(held)))
    report("Held-back, steps 1 to 3", run(clf, data[cut:], False))
    report("Held-back, with the rule check", run(clf, data[cut:], True))

    if "--dev" in sys.argv:          # while developing the rules, stop here: set C stays unseen
        sys.exit(0)

    # the real test: train on everything, run the unseen set once
    clf = train(labelled(data))
    test = json.load(open(os.path.join(HERE, "tasks_third.json")))
    say("")
    say("Unseen set C (150 requests, new accounts, reworded by a different model):")
    report("Set C, steps 1 to 3", run(clf, test, False, "classic__Lc-c.jsonl"))
    report("Set C, with the rule check", run(clf, test, True, "classic__Lc4-c.jsonl"))
    say("Run time: %.1f seconds on a laptop-class CPU, no GPU, no model call." % (time.time() - t0))
    if "--no-log" not in sys.argv:
        open(os.path.join(HERE, "results", "classic_log.txt"), "w").write("\n".join(LOG) + "\n")
