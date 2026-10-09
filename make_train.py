"""Builds train_tasks.json, the fine-tuning data.

- Accounts come from seeds used in no test set, so no customer, ID or amount repeats.
- Three in four requests are reworded, in four styles, by Gemini 2.5 Flash-Lite. The test
  sets were reworded by a different model (Claude Haiku 4.5) in one casual style.
- Each example stores the request and the complete correct answer.

   python3 make_train.py        (resumable; needs the gateway token in .env)
"""
import concurrent.futures as cf
import json
import os
import random
import re
import time

import make_tasks as m
from stack import HERE, SPECS, gateway_post

SEEDS = [310001, 310002, 310003, 310004, 310005, 310006, 310007]
OUT = os.path.join(HERE, "train_tasks.json")
STYLES = {
    "casual": "the way a busy shop owner types in a chat app: lower case, short, informal",
    "formal": "as a polite, complete business email sentence or two",
    "terse": "as a clipped note with as few words as possible",
    "plain": "in simple English as written by someone for whom English is a second language, with different word order",
}
ASK = """Rewrite this message from an online shop owner to their assistant, {style}.

Rules:
- Keep every request and its meaning exactly. Do not add information and do not drop any request.
- Keep every person's full name, every ID, tracking number, amount, quantity, quoted product name and date expression exactly as written.
- Do not answer the message.

Reply with only the rewritten message.

Message: {text}"""


def complete(call, original):
    """The answer key leaves out free-text fields; a training answer needs them filled in."""
    args = {}
    for k, v in call["args"].items():
        if isinstance(v, dict):
            continue                      # optional field that should be left out
        args[k] = v
    if call["tool"] == "create_invoice":
        g = re.search(r"for \d+ x (.+?) at \$", original) or re.search(r": \d+ (.+?), \$", original)
        args["item_name"] = g.group(1) if g else "service"
    if call["tool"] == "cancel_subscription":
        args["reason"] = "Customer asked to cancel"
    if call["tool"] == "accept_dispute_claim":
        args["note"] = "Claim accepted"
    order = list(SPECS[call["tool"]]["props"])
    return {"tool": call["tool"], "args": {k: args[k] for k in order if k in args}}


def must_keep(text, acct):
    names = {r["customer"] for c in ("orders", "invoices", "disputes", "subscriptions") for r in acct[c]}
    keep = [n for n in names if n in text]
    keep += re.findall(r"(?<![.!?:] )(?<!^)\b[A-Z][a-z]+ [A-Z][a-z]+\b", text)
    keep += re.findall(r"\d+(?:\.\d+)?", text)
    keep += re.findall(r"\b[A-Z0-9][A-Z0-9-]{7,}\b", text)
    keep += re.findall(r'"([^"]+)"', text)
    return list(dict.fromkeys(keep))


KEYWORDS = ["physical", "digital", "service", "cash", "check", "bank transfer", "ups", "fedex", "dhl", "usps", "draft", "paid",
            "unpaid", "resolved", "under review", "cancel", "refund", "remind", "yesterday", "today", "september", "october",
            "full", "qr", "tracking", "dispute", "subscription", "invoice", "product", "transaction", "plans"]


def faithful(old, new):
    """Rejects a rewording that adds a number or drops a word the answer depends on."""
    nums = lambda x: set(re.findall(r"\d+(?:\.\d+)?", x))
    o, n = old.lower(), new.lower()
    return not (nums(n) - nums(o)) and not any(k in o and k not in n for k in KEYWORDS)


def reword(t, style):
    keep = must_keep(t["request"], t["account"])
    for _ in range(2):
        try:
            d = gateway_post("gemini/v1beta/openai/chat/completions",
                             {"model": "gemini-2.5-flash-lite", "temperature": 0.9, "max_tokens": 300,
                              "messages": [{"role": "user", "content": ASK.format(style=STYLES[style], text=t["request"])}]})
            new = (d["choices"][0]["message"].get("content") or "").strip().strip('"')
        except Exception as e:
            print("  error", t["id"], str(e)[:120]); time.sleep(3); continue
        if new and all(k.lower() in new.lower() for k in keep) and new.lower() != t["request"].lower() and faithful(t["request"], new):
            return new
    return None


def source():
    out, pick = [], random.Random(99)
    for seed in SEEDS:
        m.R.seed(seed)
        for t in m.build():
            t = dict(t); t["id"] = "R%04d" % (len(out) + 1)
            t["request_original"] = t["request"]
            t["answer"] = [complete(c, t["request"]) for c in t["expected"]]
            t["style"] = pick.choice(list(STYLES) * 3 + ["original"] * 4)
            out.append(t)
    return out


if __name__ == "__main__":
    src = source()
    done = {t["id"]: t for t in json.load(open(OUT))} if os.path.exists(OUT) else {}
    todo = [t for t in src if t["id"] not in done]
    start = time.time()

    def work(t):
        if time.time() - start > 140:
            return
        if t["style"] != "original":
            new = reword(t, t["style"])
            if new:
                t["request"] = new
            else:
                t["style"] = "original"
        done[t["id"]] = t

    with cf.ThreadPoolExecutor(6) as ex:
        list(ex.map(work, todo))
    json.dump([done[t["id"]] for t in src if t["id"] in done], open(OUT, "w"))
    from collections import Counter
    print("done %d/%d |" % (len(done), len(src)), dict(Counter(t["style"] for t in done.values())))
