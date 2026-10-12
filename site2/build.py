"""Builds the two client pages from recorded results.

  python3 site2/build.py <payments repo dir> <finqa repo dir>

Each page: three examples in context, the ladder, the rules with their measured effect, the fine-tuning story,
what did not help, and every test item with its recorded answers, which the page replays on click.
Nothing is run live: every answer shown was recorded in results/ or fin/results/.
"""
import collections
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from score import call_ok  # noqa: E402

TEMPLATE = open(os.path.join(ROOT, "site2", "template.html")).read()
WRITE = {"create_invoice", "send_invoice", "send_invoice_reminder", "cancel_sent_invoice", "record_payment_for_invoice",
         "create_refund", "accept_dispute_claim", "create_shipment_tracking", "cancel_subscription", "create_product"}


def jl(path):
    return {json.loads(l)["id"]: json.loads(l) for l in open(path)}


def pct(rs, ids, key):
    return round(100.0 * sum(bool(rs[i][key]) for i in ids) / len(ids))


def loss_points(path):
    d = json.load(open(os.path.join(ROOT, path)))
    pts = [[x["step"], round(x["loss"], 4)] for x in d["log"] if "loss" in x and "step" in x]
    return pts, d.get("minutes"), d.get("examples")


# ----------------------------------------------------------------------------------------------------------- payments
def show_call(c):
    args = []
    for k, v in (c.get("args") or {}).items():
        if isinstance(v, dict):
            if "$absent" in v:
                continue
            v = v.get("$absent_or")
        if v in (None, "") or k in ("note", "note_to_payer", "reason", "item_name"):
            continue
        args.append("%s=%s" % (k, v))
    return "%s(%s)" % (c["tool"], ", ".join(args))


def shown(calls):
    return [show_call(c) for c in calls] or ["(no action: stops and asks the merchant)"]


def records_for(task, extra_calls):
    """The few account records a reader needs to judge this request: those named by ID in an answer, or by customer in the request."""
    acct, text = task["account"], task["request"].lower()
    idk = ("order_id", "capture_id", "invoice_id", "dispute_id", "subscription_id", "product_id")
    ids = {str(v).lower() for c in task["expected"] + extra_calls for k, v in (c.get("args") or {}).items() if k in idk and isinstance(v, str)}
    out = []
    for coll in ("orders", "invoices", "disputes", "subscriptions"):
        for r in acct.get(coll, []):
            name = str(r.get("customer", "")).lower()
            hit = any(str(r.get(k, "")).lower() in ids for k in idk if r.get(k)) or (name and name in text)
            if hit:
                rid = r.get({"orders": "order_id", "invoices": "invoice_id", "disputes": "dispute_id", "subscriptions": "subscription_id"}[coll])
                what = r.get("item") or r.get("reason") or r.get("plan") or ""
                amt = r.get("amount")
                out.append({"kind": coll[:-1], "id": rid, "pay": r.get("capture_id", "") if coll == "orders" else "", "customer": r.get("customer", ""), "what": what,
                            "amount": "" if amt is None else "$%g" % amt, "status": r.get("status") or r.get("state", "")})
    return out[:4]


def payments(out_dir):
    tasks = {t["id"]: t for t in json.load(open(os.path.join(ROOT, "tasks_third.json")))}
    ids = list(tasks)
    alone = jl(os.path.join(ROOT, "results", "bedrock-google.gemma-3-4b-it__L0-t.jsonl"))
    rules = jl(os.path.join(ROOT, "results", "bedrock-google.gemma-3-4b-it__L1234-u.jsonl"))
    tuned = jl(os.path.join(ROOT, "results", "hf-gemma-3-4b-tuned__L1234-ft.jsonl"))
    base_c = jl(os.path.join(ROOT, "results", "hf-gemma-3-4b-base__L0-ft.jsonl"))
    tuned_alone = jl(os.path.join(ROOT, "results", "hf-gemma-3-4b-tuned__L0-ft.jsonl"))

    def wrong(r, t):
        return any(c["tool"] in WRITE and not any(call_ok(c, g) for g in t["expected"]) for c in r["calls"])

    def nwrong(rs):
        return sum(wrong(rs[i], tasks[i]) for i in ids)

    cats = {"1_direct": "ID given in the request", "2_lookup": "Customer named, record must be found", "3_list_create": "Lists, dates, new invoices",
            "4_multi_step": "Two or three requests in one message", "5_should_not_act": "Should stop and ask"}
    items = []
    for i in ids:
        t = tasks[i]
        rej = list(dict.fromkeys(rules[i].get("rejections") or []))[:2]
        items.append({"id": i, "cat": cats[t["category"]], "q": t["request"],
                      "ctx": records_for(t, rules[i]["calls"] + alone[i]["calls"]),
                      "want": shown(t["expected"]), "why": t.get("why_no_action", ""),
                      "a": {"alone": {"text": shown(alone[i]["calls"]), "ok": alone[i]["pass"], "bad": wrong(alone[i], t)},
                            "rules": {"text": shown(rules[i]["calls"]), "ok": rules[i]["pass"], "bad": wrong(rules[i], t), "note": rej},
                            "ftonly": {"text": shown(tuned_alone[i]["calls"]), "ok": tuned_alone[i]["pass"], "bad": wrong(tuned_alone[i], t)},
                            "tuned": {"text": shown(tuned[i]["calls"]), "ok": tuned[i]["pass"], "bad": wrong(tuned[i], t)}}})

    def pick(test):
        return next(x["id"] for x in items if test(x))
    sig = lambda x: "".join("1" if x["a"][k]["ok"] else "0" for k in ("alone", "rules", "ftonly", "tuned"))
    ex = [{"id": pick(lambda x: sig(x) == "0101" and x["a"]["alone"]["bad"] and "REFUNDED" in " ".join(x["a"]["rules"]["note"])),
           "label": "Rules fixed it, fine-tuning alone did not"},
          {"id": pick(lambda x: sig(x) == "0011" and x["cat"] == cats["4_multi_step"]),
           "label": "Fine-tuning fixed it, rules alone did not"},
          {"id": pick(lambda x: sig(x) == "0001"), "label": "Only the two together got it right"}]

    # rules, each switched off on its own (Gemma 3 4B, all four layers otherwise on), recorded in results/*-r-*.jsonl
    R = lambda tag: jl(os.path.join(ROOT, "results", "bedrock-google.gemma-3-4b-it__L1234-r-%s.jsonl" % tag))
    allr = R("all")
    base_right, base_wrong = sum(allr[i]["pass"] for i in ids), nwrong(allr)
    catalog = [
        ("state", "Status must allow the action", "Refund only completed orders; send only draft invoices; remind or cancel only sent ones; cancel only active subscriptions; accept only disputes awaiting action.", "off-state"),
        ("ambiguous", "Ask when two records fit", "If the customer has more than one record the action could apply to, stop and ask which.", "off-ambiguous"),
        ("given", "Typed details must come from the message", "A tracking number, refund ID or product name must appear in what the merchant wrote.", "off-given"),
        ("amount", "Amounts must be stated", "A partial refund, payment or invoice price must be a number the merchant wrote.", "off-amount"),
        ("exists", "The record must exist", "Every ID must belong to a record on this account.", "off-exists"),
        ("schema", "The action must be well formed", "The tool exists, required fields are present, values are allowed.", "off-schema"),
        ("customer", "Invoices go to known customers", "A new invoice's email must belong to a customer on the account.", "off-customer"),
        ("dates", "Dates must be valid", "Date ranges in order and not after today.", "off-dates"),
    ]
    rule_rows = []
    for key, name, desc, tag in catalog:
        r = R(tag)
        rule_rows.append({"name": name, "desc": desc, "kind": "kept",
                          "pts": base_right - sum(r[i]["pass"] for i in ids), "wrong": nwrong(r) - base_wrong})
    for tag, name, desc in [("add-explicit_id", "Act only on IDs the merchant typed", "Refuse any money-moving action unless the record's ID is in the message."),
                            ("add-strict_lookup", "Look-ups only on open records", "Refuse tracking on refunded orders and QR codes on paid invoices.")]:
        r = R(tag)
        rule_rows.append({"name": name, "desc": desc, "kind": "tried",
                          "pts": sum(r[i]["pass"] for i in ids) - base_right, "wrong": base_wrong - nwrong(r)})

    loss, minutes, n_ex = loss_points("results/finetune_training_log.json")
    cat_rows = []
    for k, label in cats.items():
        ii = [i for i in ids if tasks[i]["category"] == k]
        cat_rows.append({"label": label, "n": len(ii), "base": pct(base_c, ii, "pass"), "tuned": pct(tuned_alone, ii, "pass"),
                         "alone": pct(alone, ii, "pass"), "rules": pct(rules, ii, "pass"), "both": pct(tuned, ii, "pass")})

    data = {
        "kind": "payments", "n": len(ids),
        "setups": [{"key": "alone", "name": "Small model alone"}, {"key": "rules", "name": "Rules only"}, {"key": "ftonly", "name": "Fine-tuning only"}, {"key": "tuned", "name": "Both: final"}],
        "ladder": [{"name": "Small model alone", "pct": pct(alone, ids, "pass"), "wrong": nwrong(alone)},
                   {"name": "Rules only", "pct": pct(rules, ids, "pass"), "wrong": nwrong(rules)},
                   {"name": "Fine-tuning only", "pct": pct(tuned_alone, ids, "pass"), "wrong": nwrong(tuned_alone)},
                   {"name": "Both: final", "pct": pct(tuned, ids, "pass"), "wrong": nwrong(tuned), "final": True}],
        "rules": rule_rows, "base_right": base_right, "examples": ex, "items": items,
        "ft": {"minutes": round(minutes), "examples": n_ex, "loss": loss, "cats": cat_rows,
               "alone_base": pct(base_c, ids, "pass"), "alone_tuned": pct(tuned_alone, ids, "pass")},
    }
    copy = {
        "title": "Small model, safe payment actions",
        "eyebrow": "Payments operations",
        "h1": "A small model that turns merchant requests into payment actions, made reliable with rules and fine-tuning",
        "lead": "A merchant writes a request in plain words. The model must pick the right action on the account, fill in the right record, and stop when the account does not allow it. Tested on %d requests it had never seen." % len(ids),
        "ex_h": "The task, in three requests",
        "ex_p": "One request the rules fixed, one that fine-tuning fixed, and one that needed both. Each card shows the account records involved, what the merchant wrote and the correct action. Press Play to replay what each setup did.",
        "ladder_h": "What moved the score",
        "ladder_note": "Each bar is a separate setup tested on all %d requests; the last combines the two before it. Share handled exactly right; the line under each bar counts wrong money-moving actions: refunds, cancellations and invoices that should not have happened." % len(ids),
        "steps": [
            ["Small model alone", "Gemma 3 4B, an open model small enough to run on one ordinary GPU, reads the request and the account and replies with the actions."],
            ["Rules only", "No training. Code around the model checks every reply: the reply must be well formed, IDs must come from the account, a message with several requests is split up, and each action is tested against the account before it runs. The model gets one chance to correct itself."],
            ["Fine-tuning only", "The same model trained on 800 example requests with their correct actions, on a free cloud GPU, with no rules around it. It gets more right, but still makes %d wrong money-moving actions: it acts when it should stop." % nwrong(tuned_alone)],
            ["Both: the final approach", "The fine-tuned model with the rules on. The highest score, and wrong actions fall to %d. The rules alone had %d: the trained model acts more readily, and a few of its actions pass every rule." % (nwrong(tuned), nwrong(rules))]],
        "rules_h": "Rules: which ones earned their place",
        "rules_p": "Each rule was switched off on its own and the full test re-run, so its number is what the score loses without it. Rules we tried and dropped were added on their own the same way.",
        "ft_h": "Fine-tuning: what it fixed and what it did not",
        "ft_insight": "Fine-tuning taught the model the shape of the job: multi-part messages went from %d%% to %d%%. It did not teach the model when to refuse. Requests that should be refused rose only from %d%% to %d%%, and the rules take them to %d%%." % (
            cat_rows[3]["base"], cat_rows[3]["tuned"], cat_rows[4]["base"], cat_rows[4]["tuned"], cat_rows[4]["both"]),
        "ft_cols": ["Type of request", "Requests", "Untuned", "Fine-tuned", "Fine-tuned + rules"],
        "nohelp_h": "Tried and dropped",
        "nohelp": [
            ["Splitting messages on joining words", "Splitting on words such as “also” and “next” scored 97% on our first test set and fell to 86% on a reworded one. It was replaced by splitting on the records each part names."],
            ["Act only on IDs the merchant typed", "Safer on paper, but most real requests name a customer, not an ID. Correct answers fell by %d points." % -rule_rows[8]["pts"]],
            ["Stricter look-up rules", "Blocking look-ups on refunded orders and paid invoices changed nothing on the test and refused reasonable requests in development."],
            ["Forcing the template inside the model provider", "Provider-side strict templates were rate-limited on small models, so the template is checked in code instead."]],
        "all_h": "Every request",
        "all_p": "Click a request to replay it, or run all %d at once." % len(ids),
        "col_q": "Request", "col_want": "Correct action",
        "limits": "The requests were generated for this study and reworded by a separate model so the wording is not ours. The account is a test account. Fine-tuned answers were recorded on a free cloud GPU with the model compressed to 4 bits; the others through a hosted copy of the same model. All results are single runs.",
    }
    write(out_dir, copy, data)


# ------------------------------------------------------------------------------------------------------------- finqa
def finqa(out_dir):
    F = os.path.join(ROOT, "fin")
    T = {t["id"]: t for t in json.load(open(os.path.join(F, "data", "test.json")))}
    ids = list(T)
    alone = jl(os.path.join(F, "results", "google.gemma-3-4b-it__L0-t.jsonl"))
    tuned = jl(os.path.join(F, "results", "gemma-3-4b-tuned__L2-t.jsonl"))

    def num(v, gold):
        if v is None:
            return "no answer"
        if isinstance(v, str):
            return v
        if isinstance(gold, float) and abs(gold) < 1 and abs(v) < 1:
            return "%.2f%%" % (v * 100)
        return "%.4g" % v

    def gold_text(g):
        return num(g, g)

    def tail(raw):
        raw = (raw or "").strip()
        lines = [l for l in raw.splitlines() if l.strip()]
        return lines[-4:-1] if len(lines) > 1 else []

    items = []
    for i in ids:
        t = T[i]
        a, u = alone[i], tuned[i]
        items.append({"id": i, "cat": {1: "One step", 2: "Two steps"}.get(t["steps"], "Three or more steps"),
                      "q": t["question"], "table": t["table"][:12], "before": (lambda b: b if len(b) > 60 else "")(" ".join(t["before"])[-280:]),
                      "want": [gold_text(t["answer"])], "formula": t.get("formula") or "",
                      "a": {"alone": {"text": [l[:160] for l in tail(a.get("raw"))] + ["Answer: %s" % num(a["answer"], t["answer"]) if not isinstance(a["answer"], str) else "Answer: %s" % a["answer"]], "ok": a["right"], "bad": False},
                            "tuned": {"text": ["Formula: %s" % (u.get("formula") or "none"), "Calculator: %s" % num(u["answer"], t["answer"])], "ok": u["right"], "bad": False}}})

    def small(x):
        tb = T[x["id"]]["table"]
        return 3 <= len(tb) <= 6 and len(tb[0]) <= 4 and len(x["q"]) < 120
    ex = []
    for test in [lambda x: x["cat"] == "Two steps" and not x["a"]["alone"]["ok"] and x["a"]["tuned"]["ok"],
                 lambda x: x["cat"] == "One step" and not x["a"]["alone"]["ok"] and x["a"]["tuned"]["ok"],
                 lambda x: x["cat"] == "Two steps" and x["a"]["alone"]["ok"] and not x["a"]["tuned"]["ok"]]:
        ex.append(next(x["id"] for x in items if small(x) and x["id"] not in ex and test(x)))
    ex = [{"id": e, "label": l} for e, l in zip(ex, ["Fine-tuning fixed it: the model's own arithmetic slipped", "Fine-tuning fixed it: the model gave the amount, not the share asked for", "Fine-tuning made it worse"])]

    dev = lambda f: round(100.0 * sum(r["right"] for r in jl(os.path.join(F, "results", f + ".jsonl")).values()) / 100)
    base_dev = dev("google.gemma-3-4b-it__L0-d")
    tried = [("Calculator redoes the arithmetic", "The model writes its working as a formula; code computes it.", "google.gemma-3-4b-it__L1-d3"),
             ("Every figure must be printed in the report", "Numbers in the formula that do not appear in the extract are sent back once.", "google.gemma-3-4b-it__L12-d3"),
             ("Calculator, figure check and worked examples", "The two above plus four similar solved questions from the training set.", "google.gemma-3-4b-it__L123-d3"),
             ("Worked examples only", "Four similar solved questions shown before the extract.", "google.gemma-3-4b-it__L3-d4"),
             ("Five attempts, most common answer", "Ask five times and keep the answer given most often.", "google.gemma-3-4b-it__L4-d4"),
             ("Table rewritten row by row", "Each table cell written out with its row and column names.", "google.gemma-3-4b-it__L5-d5")]
    rule_rows = [{"name": n, "desc": d, "kind": "tried", "pts": dev(f) - base_dev, "wrong": None} for n, d, f in tried]

    loss, minutes, n_ex = loss_points("fin/results/finetune_training_log.json")
    cat_rows = []
    for lab in ["One step", "Two steps", "Three or more steps"]:
        ii = [x["id"] for x in items if x["cat"] == lab]
        cat_rows.append({"label": lab, "n": len(ii), "base": pct(alone, ii, "right"), "tuned": pct(tuned, ii, "right")})
    data = {
        "kind": "finqa", "n": len(ids),
        "setups": [{"key": "alone", "name": "Small model alone"}, {"key": "tuned", "name": "Fine-tuned"}],
        "ladder": [{"name": "Small model alone", "pct": pct(alone, ids, "right")}, {"name": "Rules only", "pct": None},
                   {"name": "Final: fine-tuned", "pct": pct(tuned, ids, "right"), "final": True}],
        "rules": rule_rows, "base_dev": base_dev, "examples": ex, "items": items,
        "ft": {"minutes": round(minutes), "examples": n_ex, "loss": loss, "cats": cat_rows},
    }
    three = cat_rows[2]
    copy = {
        "title": "Small model, financial reports",
        "eyebrow": "Financial statement analysis",
        "h1": "A small model that works out figures from annual reports, and what it takes to improve it",
        "lead": "The model reads one page of a bank's, insurer's or payment company's annual report and calculates what the analyst asks: a growth rate, a share of a total, a change between years. %d questions from FinQA, a public benchmark written by finance experts, none seen in training." % len(ids),
        "ex_h": "The task, in three questions",
        "ex_p": "Each card shows the table from the report, the question and the benchmark's answer. Press Play to replay what each setup answered.",
        "ladder_h": "What moved the score",
        "ladder_note": "Each bar is a separate setup tested on all %d questions. Share answered correctly, to within 1%%." % len(ids),
        "steps": [
            ["Small model alone", "Gemma 3 4B, an open model small enough to run on one ordinary GPU, reads the page, works step by step and gives a number."],
            ["Rules only", "Six code-only fixes were tried on 100 practice questions: a calculator, a check that figures are printed in the report, worked examples, voting and others. None beat the model alone, so none is used."],
            ["Final: fine-tuned", "The same model trained on %d solved questions from the benchmark's training set to write the formula; code computes it and checks every figure is printed in the report. That one check is kept because it costs nothing with the fine-tuned model, which writes formulas from printed figures." % n_ex]],
        "rules_h": "Rules: none of them helped here",
        "rules_p": "Each was tried on 100 practice questions from the benchmark's separate development set. The number is its change against the model alone (%d%%). The mistakes are mostly the wrong row, the wrong year or a misread question, which a calculator cannot repair. A few points either way is within chance at this size." % base_dev,
        "ft_h": "Fine-tuning: better on short questions, worse on long ones",
        "ft_insight": "Fine-tuning taught the model to write one compact formula. That helped one- and two-step questions, but on questions needing three or more steps it went from %d%% to %d%%: the trained model writes a formula that is too short and stops. Training examples with longer working are the next thing to try." % (three["base"], three["tuned"]),
        "ft_cols": ["Steps in the answer", "Questions", "Alone", "Fine-tuned"],
        "nohelp_h": "Tried and dropped",
        "nohelp": [
            ["A formula with no working", "Asking for the formula only, without reasoning first, cut the score from %d%% to %d%% on the practice questions. The model needs to reason before it commits to a formula." % (base_dev, dev("google.gemma-3-4b-it__L1-d2"))],
            ["A formula with no working, plus the figure check", "Adding the check that figures are printed in the report did not recover it: %d%%." % dev("google.gemma-3-4b-it__L12-d2")],
            ["Showing the right page only", "Not tried here: the benchmark already gives the model the right page. In a full annual report, finding the page is its own problem."]],
        "all_h": "Every question",
        "all_p": "Click a question to replay it, or run all %d at once." % len(ids),
        "col_q": "Question", "col_want": "Correct answer",
        "limits": "The benchmark gives the model the right page; finding that page in a full report is not tested. The benchmark's own answers contain some errors, which caps every score. A share may be written as 0.07 or 7%. Fine-tuned answers were recorded on a free cloud GPU with the model compressed to 4 bits. All results are single runs.",
    }
    write(out_dir, copy, data)


def write(out_dir, copy, data):
    page = TEMPLATE
    for k, v in copy.items():
        if isinstance(v, str):
            page = page.replace("{{%s}}" % k, v)
    page = page.replace("/*COPY*/", json.dumps(copy, ensure_ascii=False).replace("</", "<\\/"))
    page = page.replace("/*DATA*/", json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/"))
    assert "{{" not in page, page[page.index("{{"):][:80]
    os.makedirs(out_dir, exist_ok=True)
    open(os.path.join(out_dir, "index.html"), "w").write(page)
    print(out_dir, len(page) // 1024, "KB", "ladder", data["ladder"], "examples", data["examples"])


if __name__ == "__main__":
    payments(sys.argv[1])
    finqa(sys.argv[2])
