"""Runs one model over the chores with one prompt and scores it.

  python3 run_eval.py gemini:gemma-3-4b-it baseline
  python3 run_eval.py gemini:gemma-3-4b-it baseline --limit 10

Resumable: chores already answered are skipped. Results go to results/<model>__<prompt>.jsonl
The the gateway key is read from .env and is never printed.
"""
import argparse
import concurrent.futures as cf
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request

from score import grade, parse_calls

def _env_file(name):
    try:
        for line in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")).read().splitlines():
            if line.strip().startswith(name + "="):
                return line.split("=", 1)[1].strip()
    except OSError:
        pass
    return ""


BASE = (os.environ.get("GATEWAY_BASE") or _env_file("GATEWAY_BASE")).rstrip("/") + "/"   # your model gateway, set in .env
ROUTES = {"gemini": "gemini/v1beta/openai/chat/completions", "groq": "groq/openai/v1/chat/completions",
          "openrouter": "openrouter/v1/chat/completions", "openai": "openai/v1/chat/completions",
          "cerebras": "cerebras/v1/chat/completions", "anthropic": "anthropic/v1/messages",
          "azure": "azure/openai/deployments/{model}/chat/completions?api-version=2024-10-21"}


def token():
    raw = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")).read().strip()
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("GATEWAY_TOKEN="):
            return line.split("=", 1)[1].strip().strip('"')
    return raw.splitlines()[0].strip()


def openrouter_key():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    for line in open(path).read().splitlines():
        line = line.strip()
        if line.startswith("OPENROUTER_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
        if line.startswith("sk-or-"):
            return line
    raise SystemExit("No OpenRouter key in .env. Add a line: OPENROUTER_API_KEY=sk-or-v1-...")


def ask(spec, prompt, tok, max_tokens):
    route, model = spec.split(":", 1)
    if route == "ollama":  # a model running on this computer, no key and no cloud
        body = {"model": model, "stream": False, "messages": [{"role": "user", "content": prompt}],
                "options": {"temperature": 0, "num_ctx": 8192, "num_predict": max_tokens}}
        req = urllib.request.Request(os.environ.get("OLLAMA_URL", "http://localhost:11434") + "/api/chat",
                                     data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=600) as r:
            d = json.loads(r.read().decode())
        return d["message"]["content"], {"in": d.get("prompt_eval_count"), "out": d.get("eval_count")}, time.time() - t0
    url = BASE + ROUTES.get(route, "").replace("{model}", model)
    head = {"Authorization": "Bearer %s:slm-opti" % tok, "Content-Type": "application/json"}
    if route == "or":  # straight to OpenRouter with your own key
        url = "https://openrouter.ai/api/v1/chat/completions"
        head["Authorization"] = "Bearer " + openrouter_key()
    if route == "anthropic":
        head["anthropic-version"] = "2023-06-01"
        body = {"model": model, "max_tokens": max_tokens, "messages": [{"role": "user", "content": prompt}]}
    else:
        body = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0}
        body["max_completion_tokens" if route in ("openai", "azure") else "max_tokens"] = max_tokens
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=head, method="POST")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=90) as r:
        d = json.loads(r.read().decode())
    secs = time.time() - t0
    if route == "anthropic":
        text = "".join(b.get("text", "") for b in d.get("content", []))
        u = d.get("usage", {}); usage = {"in": u.get("input_tokens"), "out": u.get("output_tokens")}
    else:
        text = d["choices"][0]["message"].get("content") or ""
        u = d.get("usage", {}); usage = {"in": u.get("prompt_tokens"), "out": u.get("completion_tokens")}
    return text, usage, secs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("prompt")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=700)
    ap.add_argument("--budget", type=int, default=150, help="stop after this many seconds (resume by re-running)")
    ap.add_argument("--rpm", type=float, default=40, help="max requests per minute, to stay under rate limits")
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    tasks = json.load(open(os.path.join(here, "tasks.json")))
    if a.limit:
        step = max(1, len(tasks) // a.limit); tasks = tasks[::step][:a.limit]
    tools = json.dumps(json.load(open(os.path.join(here, "tools.json"))), indent=1)
    tmpl = open(os.path.join(here, "prompts", a.prompt + ".txt")).read()
    out = os.path.join(here, "results", re.sub(r"[^A-Za-z0-9.]+", "-", a.model) + "__" + a.prompt + ".jsonl")
    done = {}
    if os.path.exists(out):
        for line in open(out):
            r = json.loads(line); done[r["id"]] = r
    todo = [t for t in tasks if t["id"] not in done]
    local = a.model.startswith("ollama:")
    if local:
        a.workers, a.rpm, a.budget = 1, 600, 10 ** 6
    tok = "" if (local or a.model.startswith("or:")) else token(); lock = threading.Lock(); start = time.time(); errs = [0]; stop = threading.Event()
    gate = threading.Lock(); nxt = [0.0]

    def pace():
        with gate:
            wait = nxt[0] - time.time()
            nxt[0] = max(nxt[0], time.time()) + 60.0 / a.rpm
        if wait > 0:
            time.sleep(wait)

    def work(t):
        if stop.is_set() or time.time() - start > a.budget:
            return
        prompt = tmpl.replace("{tools}", tools).replace("{account}", json.dumps(t["account"], indent=1)).replace("{request}", t["request"])
        for attempt in range(3):
            try:
                pace()
                text, usage, secs = ask(a.model, prompt, tok, a.max_tokens)
                break
            except urllib.error.HTTPError as e:
                msg = "%s %s" % (e.code, e.read().decode(errors="replace")[:160].replace("\n", " "))
            except Exception as e:
                msg = repr(e)[:160]
            with lock:
                errs[0] += 1
                if errs[0] >= 3:
                    stop.set()
            if stop.is_set():
                print("  error:", msg); return
            time.sleep(8 * (attempt + 1))
        else:
            print("  gave up on", t["id"], msg); return
        calls, parsed = parse_calls(text)
        ok, why = grade(calls, t["expected"])
        rec = {"id": t["id"], "category": t["category"], "pass": ok, "reason": why, "parsed": parsed,
               "calls": calls, "raw": text[:1500], "secs": round(secs, 2), "usage": usage}
        with lock:
            errs[0] = 0
            done[t["id"]] = rec
            if local:
                n = len(done); p = sum(r["pass"] for r in done.values())
                print("  %s %-4s %3d/%d done, %d right so far (%.0fs)" % (t["id"], "ok" if ok else "FAIL", n, len(tasks), p, secs), flush=True)
            with open(out, "a") as f:
                f.write(json.dumps(rec) + "\n")

    with cf.ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(work, todo))
    rs = [done[t["id"]] for t in tasks if t["id"] in done]
    if stop.is_set():
        print("STOPPED: repeated errors from the model service.")
    print("%s | %s | answered %d/%d | passed %d (%.0f%%)" % (a.model, a.prompt, len(rs), len(tasks), sum(r["pass"] for r in rs), 100.0 * sum(r["pass"] for r in rs) / max(1, len(rs))))
    return 0 if len(rs) == len(tasks) else 2


if __name__ == "__main__":
    sys.exit(main())
