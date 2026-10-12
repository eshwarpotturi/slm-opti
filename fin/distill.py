"""Asks a large model to work through FinQA training questions step by step, and keeps only the worked answers
whose formula gives the benchmark's answer. The result teaches a small model how to reason, not just what to answer.

  python3 fin/distill.py <FinQA train.json> --n 3000

Writes fin/data/distill.jsonl (resumable). Training questions only; no test question is ever sent.
"""
import argparse
import concurrent.futures as cf
import json
import os
import random
import re
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import prep  # noqa: E402
import run  # noqa: E402
import stack  # noqa: E402

BIG = "bedrock:us.anthropic.claude-haiku-4-5-20251001-v1:0"
TEACH = ("You are a financial analyst. Read the extract from a company's annual report and answer the question.\n"
         "Reply in exactly this shape and nothing else:\n"
         "Figures:\n- <each figure you need, with what it is and its year, copied exactly from the extract>\n"
         "Working: <two or three short sentences on how the figures combine to answer the question>\n"
         "FORMULA: <the arithmetic, using only those figures, with + - * / and brackets>\n"
         "A percentage change is (new - old) / old. A share is part / total. Give shares and changes as decimals, not times 100.\n"
         "If the question asks yes or no, end with ANSWER: yes or ANSWER: no instead of FORMULA.")


def stamp():
    return time.strftime("%H:%M:%S")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("train"); ap.add_argument("--n", type=int, default=3000); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--budget", type=float, default=160)
    a = ap.parse_args()
    raw = json.load(open(a.train))
    pool = []
    for x in raw:
        t = prep.slim(x)
        text = json.dumps(t).lower()
        if any(w in text for w in prep.NAMES) or len(run.render(t).split()) > 900:
            continue
        if isinstance(t["answer"], str) or (t["formula"] and run._safe(t)):
            pool.append(t)
    random.Random(11).shuffle(pool)
    pool = pool[:a.n]
    out = os.path.join(HERE, "data", "distill.jsonl")
    seen = {json.loads(l)["id"] for l in open(out)} if os.path.exists(out) else set()
    todo = [t for t in pool if t["id"] not in seen]
    print("%s  starting: %d questions, %d already done, %d to do" % (stamp(), len(pool), len(seen), len(todo)), flush=True)
    lock, t0, stats = threading.Lock(), time.time(), {"kept": 0, "dropped": 0, "cost": 0.0, "errors": 0}

    def work(t):
        if time.time() - t0 > a.budget or stats["errors"] >= 5:
            return
        try:
            text, cost, _ = stack.chat(BIG, [{"role": "user", "content": TEACH + "\n\n" + run.ask(t)}], temperature=0, max_tokens=500)
            stats["errors"] = 0
        except Exception as e:
            stats["errors"] += 1
            print("%s  error %s %s" % (stamp(), t["id"], repr(e)[:150]), flush=True)
            return
        yn = re.findall(r"ANSWER\s*:\s*(yes|no)\b", text, flags=re.I)
        f = run.read_formula(text)
        ok = False
        try:
            ok = run.correct(yn[-1].lower(), t["answer"]) if yn else (f is not None and run.correct(run.calc(f), t["answer"]))
        except Exception:
            ok = False
        rec = {"id": t["id"], "keep": bool(ok), "reply": text.strip(), "cost": cost}
        with lock:
            stats["cost"] += cost
            stats["kept" if ok else "dropped"] += 1
            open(out, "a").write(json.dumps(rec) + "\n")
            n = len(seen) + stats["kept"] + stats["dropped"]
            if n % 25 == 0:
                print("%s  %d/%d done, kept %d, dropped %d, cost this run $%.2f" % (stamp(), n, len(pool), stats["kept"], stats["dropped"], stats["cost"]), flush=True)

    with cf.ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(work, todo))
    rs = [json.loads(l) for l in open(out)] if os.path.exists(out) else []
    print("%s  total %d/%d, kept %d (%.0f%%), cost so far $%.2f" % (stamp(), len(rs), len(pool), sum(r["keep"] for r in rs),
          100.0 * sum(r["keep"] for r in rs) / max(1, len(rs)), sum(r["cost"] for r in rs)), flush=True)


if __name__ == "__main__":
    main()
