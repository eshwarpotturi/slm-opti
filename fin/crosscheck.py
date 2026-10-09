"""Two small models check each other. Where their answers match, that answer is used. Where they differ, a larger model answers.

  python3 fin/crosscheck.py --tag t --big bedrock:us.anthropic.claude-haiku-4-5-20251001-v1:0

Needs both small models' --layers 0 runs for the same --tag. Writes fin/results/crosscheck-<tag>.jsonl. Resumable.
"""
import argparse
import concurrent.futures as cf
import json
import os
import threading
import time

import run

A = "google.gemma-3-4b-it"
B = "mistral.ministral-3-3b-instruct"


def same(x, y):
    if x is None or y is None:
        return False
    if isinstance(x, str) or isinstance(y, str):
        return x == y
    return abs(x - y) <= 0.01 * max(abs(x), abs(y), 1e-9)


def load(name):
    return {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(run.HERE, "results", name + ".jsonl"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="t"); ap.add_argument("--tasks", default="data/test.json")
    ap.add_argument("--big", default="bedrock:us.anthropic.claude-haiku-4-5-20251001-v1:0")
    ap.add_argument("--a", default=A + "__L0", help="result-file stem of the first small model, e.g. gemma-3-4b-tuned__L0")
    ap.add_argument("--out", default="crosscheck")
    ap.add_argument("--budget", type=float, default=160); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    tasks = json.load(open(os.path.join(run.HERE, a.tasks)))
    ra, rb = load("%s-%s" % (a.a, a.tag)), load("%s__L0-%s" % (B, a.tag))
    out = os.path.join(run.HERE, "results", "%s-%s.jsonl" % (a.out, a.tag))
    done = {json.loads(l)["id"] for l in open(out)} if os.path.exists(out) else set()
    lock, t0, errs = threading.Lock(), time.time(), [0]

    def work(t):
        i = t["id"]
        if i in done or i not in ra or i not in rb or time.time() - t0 > a.budget or errs[0] >= 3:
            return
        r = {"id": i, "gold": t["answer"], "a": ra[i]["answer"], "b": rb[i]["answer"], "cost": ra[i]["cost"] + rb[i]["cost"]}
        if same(r["a"], r["b"]):
            r.update(answer=r["a"], route="agreed")
        else:
            try:
                big = run.solve(a.big, t, "0")
                errs[0] = 0
            except Exception as e:
                errs[0] += 1
                print("error", i, repr(e)[:160], flush=True)
                return
            r.update(answer=big["answer"], route="escalated", cost=r["cost"] + big["cost"])
        r["right"] = run.correct(r["answer"], t["answer"])
        with lock:
            open(out, "a").write(json.dumps(r) + "\n")

    with cf.ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(work, tasks))
    rs = [json.loads(l) for l in open(out)] if os.path.exists(out) else []
    ag = [r for r in rs if r["route"] == "agreed"]
    es = [r for r in rs if r["route"] == "escalated"]
    print("%d/%d done | agreed %d, right %d | escalated %d, right %d | overall %d (%.0f%%) | cost $%.4f" % (
        len(rs), len(tasks), len(ag), sum(r["right"] for r in ag), len(es), sum(r["right"] for r in es),
        sum(r["right"] for r in rs), 100.0 * sum(r["right"] for r in rs) / max(1, len(rs)), sum(r["cost"] for r in rs)))


if __name__ == "__main__":
    main()
