"""Builds a fresh test: new random accounts, and every request reworded by a different model
(Claude Haiku 4.5) that has never seen the stack or its rules. Names, IDs, amounts and numbers
must survive the rewording, or the original wording is kept.
   python3 paraphrase.py"""
import concurrent.futures as cf, json, os, re, time
from stack import gateway_post, HERE
import sys
SRC, OUT = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("tasks_fresh_src.json", "tasks_fresh.json")
MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
ASK = """Rewrite this message from an online shop owner to their assistant. Write it the way a different, busy person would type it: different words, different sentence structure, casual tone.

Rules:
- Keep every request and its meaning exactly. Do not add information and do not drop any request.
- Keep every person's full name, every ID, tracking number, amount, quantity, quoted product name and date expression exactly as written.
- Do not answer the message.

Reply with only the rewritten message.

Message: """


def must_keep(text, acct):
    names = {r["customer"] for c in ("orders", "invoices", "disputes", "subscriptions") for r in acct[c]}
    keep = [n for n in names if n in text]
    keep += [m for m in re.findall(r"(?<![.!?:] )(?<!^)\b[A-Z][a-z]+ [A-Z][a-z]+\b", text)]   # any other full name, not a sentence-opening verb
    keep += re.findall(r"\d+(?:\.\d+)?", text)
    keep += re.findall(r"\b[A-Z0-9][A-Z0-9-]{7,}\b", text)
    keep += re.findall(r'"([^"]+)"', text)
    return [k for k in dict.fromkeys(keep) if k not in ("Also", "Next")]


def reword(t):
    keep = must_keep(t["request"], t["account"])
    for _ in range(2):
        try:
            d = gateway_post("bedrock/%s/converse" % MODEL, {"messages": [{"role": "user", "content": [{"text": ASK + t["request"]}]}],
                                                            "inferenceConfig": {"temperature": 0.9, "maxTokens": 300}})
        except Exception as e:
            print("  error", t["id"], str(e)[:120]); time.sleep(3); continue
        new = "".join(c.get("text", "") for c in d["output"]["message"]["content"]).strip().strip('"')
        if new and all(k.lower() in new.lower() for k in keep) and new.lower() != t["request"].lower():
            return new
    return None


src = json.load(open(os.path.join(HERE, SRC)))
out_path = os.path.join(HERE, OUT)
done = {t["id"]: t for t in json.load(open(out_path))} if os.path.exists(out_path) else {}
todo = [t for t in src if t["id"] not in done]
start = time.time()


def work(t):
    if time.time() - start > 130:
        return
    new = reword(t)
    t = dict(t); t["request_original"] = t["request"]; t["reworded"] = new is not None
    if new:
        t["request"] = new
    done[t["id"]] = t


with cf.ThreadPoolExecutor(5) as ex:
    list(ex.map(work, todo))
json.dump([done[t["id"]] for t in src if t["id"] in done], open(out_path, "w"), indent=1)
print("done %d/%d | reworded %d | kept original wording %d" % (len(done), len(src), sum(t["reworded"] for t in done.values()), sum(not t["reworded"] for t in done.values())))
