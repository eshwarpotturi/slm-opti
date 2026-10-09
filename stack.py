"""The improvement stack. Each layer is code around the model, not a change of wording.

  1  constrained decoding   the reply must match a strict template built from the tool list
  2  record grounding       code keeps only the records the request refers to; IDs can only come from those
  3  tool retrieval         a search step shows the model only the most relevant tools
  4  policy guard           rules check every proposed action against the account; one self-correction round
  5  escalation             when the small model is unsure, a big model takes the request

  python3 stack.py or:meta-llama/llama-3.2-3b-instruct --layers 1234
  python3 stack.py or:meta-llama/llama-3.2-3b-instruct --layers 12345 --big or:anthropic/claude-sonnet-5

Results go to results/<model>__L<layers>.jsonl. Resumable. Reads OPENROUTER_API_KEY from .env.
"""
import argparse
import concurrent.futures as cf
import json
import math
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request

from score import grade, parse_calls

HERE = os.path.dirname(os.path.abspath(__file__))
OR = "https://openrouter.ai/api/v1/"
EMBED_MODEL = "sentence-transformers/all-minilm-l12-v2"
SPEND_CAP_USD = float(os.environ.get("SPEND_CAP_USD", "2.60"))   # hard stop on total key usage

# which account records each ID field may point at
ID_FIELDS = {"invoice_id": ("invoices", "invoice_id"), "order_id": ("orders", "order_id"),
             "capture_id": ("orders", "capture_id"), "dispute_id": ("disputes", "dispute_id"),
             "subscription_id": ("subscriptions", "subscription_id"), "product_id": ("products", "product_id")}
COLLECTIONS = ("orders", "invoices", "disputes", "subscriptions")
READ_ONLY = {"get_order", "get_invoice", "get_dispute", "show_subscription_details", "get_shipment_tracking"}


def key():
    for line in open(os.path.join(HERE, ".env")).read().splitlines():
        if line.strip().startswith("OPENROUTER_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("No OPENROUTER_API_KEY in .env")


def http(path, body=None, tries=4):
    req = urllib.request.Request(OR + path, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Authorization": "Bearer " + key(), "Content-Type": "application/json"},
                                 method="POST" if body is not None else "GET")
    last = ""
    for i in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=45) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            last = "%s %s" % (e.code, e.read().decode(errors="replace")[:300].replace("\n", " "))
            if e.code in (400, 401, 402, 403, 404):
                break
            if e.code == 429:
                time.sleep(6 * (i + 1))
        except Exception as e:
            last = repr(e)[:200]
        time.sleep(5 * (i + 1))
    raise RuntimeError(last)


def key_usage():
    return float(http("key")["data"].get("usage") or 0)


# ---------------------------------------------------------------- tool specs
def load_specs():
    specs = {}
    for t in json.load(open(os.path.join(HERE, "tools.json"))):
        props, required = {}, []
        for k, desc in t["parameters"].items():
            p = {"type": "number" if desc.startswith("number") else "string"}
            m = re.search(r"One of ([A-Z_, ]+)", desc)
            if m:
                p["enum"] = [x.strip() for x in m.group(1).split(",") if x.strip()]
            if "required" in desc.split(".")[0]:
                required.append(k)
            props[k] = p
        specs[t["name"]] = {"props": props, "required": required, "raw": t}
    return specs


SPECS = load_specs()


# ---------------------------------------------------------------- layer 2: record grounding
def ground(account, request):
    """Keep only the records the request points at: by customer name or by an ID written in the request."""
    low = request.lower()
    names = {r["customer"] for c in COLLECTIONS for r in account[c]}
    hit = {n for n in names if n.lower() in low}
    if not hit:                       # allow a first name or surname on its own, if it is unambiguous
        for part_i in (0, -1):
            cand = {n for n in names if re.search(r"\b%s\b" % re.escape(n.split()[part_i].lower()), low)}
            if len(cand) == 1:
                hit |= cand
    out = {"today": account["today"]}
    for c in COLLECTIONS:
        out[c] = [r for r in account[c] if r["customer"] in hit
                  or any(k.endswith("_id") and str(v) in request for k, v in r.items())]
    out["products"] = [p for p in account["products"] if p["product_id"] in request or p["name"].lower() in low]
    return out


# ---------------------------------------------------------------- layer 3: tool retrieval
EMBED_LOCAL = False
EMBED_BACKEND = os.environ.get("EMBED_BACKEND", "gateway")
_emb_lock = threading.Lock()
_emb = None


def embed(texts):
    global _emb
    path = os.path.join(HERE, "emb_cache_gemini.json" if EMBED_BACKEND == "gateway" else "emb_cache_local.json" if EMBED_LOCAL else "emb_cache.json")
    with _emb_lock:
        if _emb is None:
            _emb = json.load(open(path)) if os.path.exists(path) else {}
            pub = os.path.join(HERE, "embeddings_public.json")      # shipped with the repository, so layer 3 runs without a gateway
            if os.path.exists(pub):
                _emb.update(json.load(open(pub)))
        need = [t for t in dict.fromkeys(texts) if t not in _emb]
        for i in range(0, len(need), 64):
            if EMBED_BACKEND == "gateway":
                d = gateway_post("gemini/v1beta/openai/embeddings", {"model": "gemini-embedding-001", "input": need[i:i + 32][:32], "dimensions": 256})
                rows = [r["embedding"][:256] for r in sorted(d["data"], key=lambda x: x.get("index", 0))]
                if len(rows) < len(need[i:i + 64]):
                    d = gateway_post("gemini/v1beta/openai/embeddings", {"model": "gemini-embedding-001", "input": need[i + 32:i + 64], "dimensions": 256})
                    rows += [r["embedding"][:256] for r in sorted(d["data"], key=lambda x: x.get("index", 0))]
            elif EMBED_LOCAL:
                rows = local_post("/api/embed", {"model": "all-minilm", "input": need[i:i + 64]})["embeddings"]
            else:
                d = http("embeddings", {"model": EMBED_MODEL, "input": need[i:i + 64]})
                rows = [r["embedding"] for r in sorted(d["data"], key=lambda x: x.get("index", 0))]
            for t, row in zip(need[i:i + 64], rows):
                _emb[t] = [round(x, 5) for x in row]
            json.dump(_emb, open(path, "w"))
        return [_emb[t] for t in texts]


def cos(a, b):
    return sum(x * y for x, y in zip(a, b)) / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)) + 1e-9)


def tool_doc(name):
    return name.replace("_", " ") + ". " + SPECS[name]["raw"]["description"]


def _has_anchor(piece, account):
    """Does this piece name something to act on: a customer, an ID, or a quoted name?"""
    low = piece.lower()
    names = {r["customer"] for c in COLLECTIONS for r in account[c]}
    if any(n.lower() in low for n in names):
        return True
    for part_i in (0, -1):
        if len({n for n in names if re.search(r"\b%s\b" % re.escape(n.split()[part_i].lower()), low)}) == 1:
            return True
    return bool(re.search(r'\b(?=[A-Za-z0-9-]*\d)[A-Za-z0-9-]{8,}\b|"[^"]+"', piece))


def segments(request, account):
    """Splits a message into its separate chores without relying on particular words.
    Pieces are cut at sentence punctuation; a piece that names nothing to act on
    (no customer, ID or quoted name) is joined to the piece before it."""
    parts = [p.strip() for p in re.split(r"(?<=[.!?:;])\s+|\n+", request) if p and p.strip()]
    out = []
    for p in parts:
        if out and not _has_anchor(p, account):
            out[-1] += " " + p
        elif len(out) == 1 and not _has_anchor(out[0], account):
            out[0] += " " + p                       # an opening piece with nothing to act on joins the next one
        else:
            out.append(p)
    return out or [request]


def sentences(text):
    return [p.strip() for p in re.split(r"(?<=[.!?:;])\s+|\n+", text) if p and len(p.strip()) > 3] or [text]


def retrieve_one(text, k=5):
    """The k most relevant tools for the chore, plus the top 3 for each sentence inside it."""
    names = list(SPECS)
    tv = embed([tool_doc(n) for n in names])
    chosen = []
    for q, top in [(text, k)] + ([(x, 3) for x in sentences(text)] if len(sentences(text)) > 1 else []):
        sv = embed([q])[0]
        chosen += [names[i] for i in sorted(range(len(names)), key=lambda i: -cos(sv, tv[i]))[:top]]
    return list(dict.fromkeys(chosen))


def retrieve(request, per_segment=5):
    names = list(SPECS)
    tv = embed([tool_doc(n) for n in names])
    segs = sentences(request)
    chosen = []
    for sv in embed(segs):
        ranked = sorted(range(len(names)), key=lambda i: -cos(sv, tv[i]))[:per_segment]
        chosen += [names[i] for i in ranked]
    return list(dict.fromkeys(chosen))


# ---------------------------------------------------------------- layer 1: constrained decoding
def build_schema(tools, grounded=None):
    """A JSON Schema the provider enforces while the model writes. With grounding, ID fields
    may only take IDs of the kept records; a tool with no possible ID is left out."""
    variants, usable = [], []
    for name in tools:
        spec = SPECS[name]
        props, ok = {}, True
        for k, p in spec["props"].items():
            q = dict(p)
            if grounded is not None and k in ID_FIELDS:
                coll, field = ID_FIELDS[k]
                ids = [r[field] for r in grounded[coll]]
                if not ids:
                    ok = ok and k not in spec["required"]
                    continue
                q["enum"] = ids
            if grounded is not None and k == "recipient_email":
                emails = sorted({r["email"] for c in ("orders", "invoices") for r in grounded[c]})
                if not emails:
                    ok = False
                    continue
                q["enum"] = emails
            props[k] = q
        if not ok:
            continue
        usable.append(name)
        variants.append({"type": "object", "additionalProperties": False, "required": ["tool", "args"],
                         "properties": {"tool": {"type": "string", "enum": [name]},
                                        "args": {"type": "object", "additionalProperties": False, "properties": props,
                                                 "required": [k for k in spec["required"] if k in props]}}})
    schema = {"type": "object", "additionalProperties": False, "required": ["calls"],
              "properties": {"calls": {"type": "array", "items": {"anyOf": variants} if variants else {"type": "object"}}}}
    return schema, usable


def template_errors(call, schema_tools):
    """Checks one call against the template: allowed tool, known fields, required fields, allowed values."""
    v = schema_tools.get(call["tool"])
    if v is None:
        return "%s is not an available tool here" % (call["tool"] or "(empty)")
    a = call.get("args") or {}
    for k in v["required"]:
        if a.get(k) in (None, ""):
            return "%s is required" % k
    for k, val in a.items():
        p = v["properties"].get(k)
        if p is None:
            return "%s is not a field of %s" % (k, call["tool"])
        if val in (None, ""):
            continue
        if "enum" in p and str(val) not in p["enum"]:
            return "%s must be one of: %s" % (k, ", ".join(p["enum"][:8]))
        if p["type"] == "number" and _num(val) is None:
            return "%s must be a number" % k
    return None


# ---------------------------------------------------------------- layer 4: policy guard
def _find(acct, coll, field, value):
    return next((r for r in acct[coll] if str(r[field]).lower() == str(value).lower()), None)


def _num(v):
    try:
        return float(str(v).replace("$", "").replace(",", ""))
    except Exception:
        return None


def _numbers_in(text):
    return {float(x) for x in re.findall(r"\d+(?:\.\d+)?", text)}


def _ambiguous(acct, coll, rec, request, eligible):
    """Same customer has more than one record this action could apply to, and the request does not say which."""
    same = [r for r in acct[coll] if r["customer"] == rec["customer"] and eligible(r)]
    if len(same) < 2:
        return False
    low = request.lower()
    tells = [str(rec.get(k, "")).lower() for k in ("item", "plan", "order_id", "invoice_id", "subscription_id", "dispute_id", "capture_id")]
    return not any(t and t in low for t in tells)


def guard(call, acct, request):
    """Returns None if the action is allowed, or a short reason why not."""
    t, a = call["tool"], call.get("args") or {}
    spec = SPECS.get(t)
    if not spec:
        return "there is no tool called %s" % t
    for k in spec["required"]:
        if a.get(k) in (None, ""):
            return "%s is required" % k
    for k, v in a.items():
        p = spec["props"].get(k)
        if p is None:
            return "%s is not a field of %s" % (k, t)
        if "enum" in p and v not in (None, "") and str(v).upper() not in p["enum"]:
            return "%s must be one of %s" % (k, ", ".join(p["enum"]))
    nums = _numbers_in(request)

    def state(coll, field, k, allowed, word):
        r = _find(acct, coll, field, a.get(k))
        if r is None:
            return None, "%s %s is not on this account" % (k, a.get(k))
        if allowed and r["status" if "status" in r else "state"] not in allowed:
            return r, "this %s is %s, so the action does not apply" % (word, r["status" if "status" in r else "state"])
        if allowed and _ambiguous(acct, coll, r, request, lambda x: x["status" if "status" in x else "state"] in allowed):
            return r, "%s has more than one %s this could mean; the merchant must say which" % (r["customer"], word)
        return r, None

    if t == "create_refund":
        o, why = state("orders", "capture_id", "capture_id", {"COMPLETED"}, "order")
        if why:
            return why
        if a.get("amount_value") not in (None, ""):
            amt = _num(a["amount_value"])
            if amt is None or amt <= 0 or amt > o["amount"] + 0.005:
                return "refund amount must be between 0 and the order amount %s" % o["amount"]
            if amt not in nums and abs(amt - o["amount"]) > 0.005:
                return "the merchant did not state this refund amount"
    elif t in ("send_invoice", "send_invoice_reminder", "cancel_sent_invoice", "record_payment_for_invoice", "generate_invoice_qr_code"):
        allowed = {"send_invoice": {"DRAFT"}, "generate_invoice_qr_code": None}.get(t, {"SENT"})
        v, why = state("invoices", "invoice_id", "invoice_id", allowed, "invoice")
        if why:
            return why
        if t == "record_payment_for_invoice":
            amt = _num(a.get("amount_value"))
            if amt is None or amt <= 0 or amt > v["amount"] + 0.005:
                return "payment amount must be between 0 and the invoice amount %s" % v["amount"]
            if amt not in nums and abs(amt - v["amount"]) > 0.005:
                return "the merchant did not state this payment amount"
    elif t == "cancel_subscription":
        _, why = state("subscriptions", "subscription_id", "subscription_id", {"ACTIVE"}, "subscription")
        if why:
            return why
    elif t == "accept_dispute_claim":
        _, why = state("disputes", "dispute_id", "dispute_id", {"REQUIRED_ACTION"}, "dispute")
        if why:
            return why
    elif t == "create_shipment_tracking":
        _, why = state("orders", "order_id", "order_id", None, "order")
        if why:
            return why
        if str(a.get("tracking_number", "")).lower() not in request.lower() or not a.get("tracking_number"):
            return "the merchant did not give this tracking number"
    elif t in ("get_order", "get_shipment_tracking"):
        _, why = state("orders", "order_id", "order_id", None, "order")
        if why:
            return why
    elif t == "get_invoice":
        _, why = state("invoices", "invoice_id", "invoice_id", None, "invoice")
        if why:
            return why
    elif t == "get_dispute":
        _, why = state("disputes", "dispute_id", "dispute_id", None, "dispute")
        if why:
            return why
    elif t == "show_subscription_details":
        _, why = state("subscriptions", "subscription_id", "subscription_id", None, "subscription")
        if why:
            return why
    elif t == "show_product_details":
        _, why = state("products", "product_id", "product_id", None, "product")
        if why:
            return why
    elif t == "get_refund":
        if not a.get("refund_id") or str(a["refund_id"]).lower() not in request.lower():
            return "the merchant did not give this refund_id"
    elif t == "create_invoice":
        emails = {r["email"] for c in ("orders", "invoices") for r in acct[c]}
        if a.get("recipient_email") not in emails:
            return "recipient_email is not a customer on this account"
        for k in ("quantity", "unit_amount"):
            if _num(a.get(k)) not in nums:
                return "the merchant did not state this %s" % k
    elif t == "list_transactions":
        for k in ("start_date", "end_date"):
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(a.get(k, ""))):
                return "%s must be YYYY-MM-DD" % k
        if a["start_date"] > a["end_date"] or a["end_date"] > acct["today"]:
            return "dates must be in order and not after today (%s)" % acct["today"]
    elif t == "create_product":
        if str(a.get("name", "")).strip().lower() not in request.lower() or not a.get("name"):
            return "the merchant did not give this product name"
    return None


def drop_redundant_lookups(calls, acct):
    """A read of a record that another action in the same plan already acts on adds nothing."""
    def rec_of(c):
        a = c.get("args") or {}
        for k, (coll, field) in ID_FIELDS.items():
            if k in a:
                r = _find(acct, coll, field, a[k])
                if r:
                    return (coll, id(r))
        return None
    acted = {rec_of(c) for c in calls if c["tool"] not in READ_ONLY} - {None}
    return [c for c in calls if not (c["tool"] in READ_ONLY and rec_of(c) in acted)]


# ---------------------------------------------------------------- the pipeline
PROMPT = open(os.path.join(HERE, "prompts", "baseline.txt")).read()
OUT_OLD = 'Reply with only a JSON array of the tool calls needed, in order. Each call looks like {"tool": "<tool name>", "args": {...}}. If the request cannot be done with these tools, or you would need to ask the merchant something first, reply with [].'
OUT_NEW = 'Reply with only a JSON object {"calls": [...]} holding the tool calls needed, in order. Each call looks like {"tool": "<tool name>", "args": {...}}. If the request cannot be done with these tools, or you would need to ask the merchant something first, reply with {"calls": []}.'
assert OUT_OLD in PROMPT


LOCAL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
def _env_file(name):
    try:
        for line in open(os.path.join(HERE, ".env")).read().splitlines():
            if line.strip().startswith(name + "="):
                return line.split("=", 1)[1].strip()
    except OSError:
        pass
    return ""


GATEWAY = (os.environ.get("GATEWAY_BASE") or _env_file("GATEWAY_BASE")).rstrip("/") + "/"   # your model gateway, set in .env
# list prices per million tokens (input, output), used only to estimate cost
PRICES = {"us.meta.llama3-1-8b-instruct-v1:0": (0.22, 0.22), "google.gemma-3-4b-it": (0.04, 0.08),
          "mistral.ministral-3-3b-instruct": (0.04, 0.04), "us.anthropic.claude-sonnet-4-5-20250929-v1:0": (3.0, 15.0)}


def gateway_token():
    for line in open(os.path.join(HERE, ".env")).read().splitlines():
        line = line.strip()
        if line.startswith("GATEWAY_TOKEN="):
            return line.split("=", 1)[1].strip()
        if line and "=" not in line.split(".")[0] and not line.startswith("sk-or-"):
            return line
    raise SystemExit("No the gateway token in .env")


class Blocked(Exception):
    pass


def gateway_post(path, body, tries=3):
    if GATEWAY == "/":
        raise SystemExit("Add a line GATEWAY_BASE=<your model gateway address> to .env")
    req = urllib.request.Request(GATEWAY + path, data=json.dumps(body).encode(), method="POST",
                                 headers={"Authorization": "Bearer %s:slm-opti" % gateway_token(), "Content-Type": "application/json"})
    last = ""
    for i in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            last = "%s %s" % (e.code, e.read().decode(errors="replace")[:240].replace("\n", " "))
            if "too_many_errors" in last:
                raise Blocked(last)
            if e.code in (400, 401, 403, 404):
                break
        except Exception as e:
            last = repr(e)[:200]
        time.sleep(4 * (i + 1))
    raise RuntimeError(last)


def local_post(path, body, timeout=900):
    req = urllib.request.Request(LOCAL + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


HF_CHAT = None      # set by the fine-tuning notebook: a function (messages, temperature, max_tokens) -> text


def chat(model, messages, schema=None, temperature=0, max_tokens=700):
    if model.startswith("hf:"):          # a model loaded in this Python process, e.g. on a Colab GPU
        t0 = time.time()
        return HF_CHAT(messages, temperature, min(max_tokens, 400)), 0.0, time.time() - t0
    if model.startswith("bedrock:"):     # an open model hosted in the organisation's AWS account, reached through the gateway
        mid = model.split(":", 1)[1]
        body = {"messages": [{"role": m["role"], "content": [{"text": m["content"]}]} for m in messages],
                "inferenceConfig": {"temperature": temperature, "maxTokens": max_tokens}}
        t0 = time.time()
        d = gateway_post("bedrock/%s/converse" % mid, body)
        text = "".join(c.get("text", "") for c in d["output"]["message"]["content"])
        u = d.get("usage") or {}
        pin, pout = PRICES.get(mid, (0.2, 0.2))
        return text, (u.get("inputTokens", 0) * pin + u.get("outputTokens", 0) * pout) / 1e6, time.time() - t0
    if model.startswith("ollama:"):      # a model running on this computer: free, and the template is enforced while it writes
        body = {"model": model.split(":", 1)[1], "messages": messages, "stream": False,
                "options": {"temperature": temperature, "num_ctx": 8192, "num_predict": max_tokens}}
        if schema is not None and os.environ.get("NO_GRAMMAR") != "1":
            body["format"] = schema
        t0 = time.time()
        d = local_post("/api/chat", body)
        return d["message"]["content"], 0.0, time.time() - t0
    body = {"model": model.split(":", 1)[1], "messages": messages, "temperature": temperature,
            "max_tokens": max_tokens, "usage": {"include": True}}
    if schema is not None and os.environ.get("NATIVE_SCHEMA") == "1":   # provider-side enforcement, where a provider offers it
        body["response_format"] = {"type": "json_schema", "json_schema": {"name": "tool_calls", "strict": True, "schema": schema}}
    t0 = time.time()
    d = http("chat/completions", body)
    if "choices" not in d:
        raise RuntimeError(json.dumps(d)[:300])
    u = d.get("usage") or {}
    return (d["choices"][0]["message"].get("content") or ""), float(u.get("cost") or 0), time.time() - t0


def repair_json(text):
    """Fixes the common slips of small models: code fences, missing quotes around field names, trailing commas."""
    t = re.sub(r"```(?:json)?", "", text or "").strip()
    i = min([x for x in (t.find("{"), t.find("[")) if x != -1] or [0])
    j = max(t.rfind("}"), t.rfind("]"))
    t = t[i:j + 1] if j > i else t
    try:
        json.loads(t)
        return t
    except Exception:
        pass
    t2 = re.sub(r'([{,\[]\s*)"?([A-Za-z_]\w*)"?\s*:', r'\1"\2":', t)
    t2 = re.sub(r",\s*([}\]])", r"\1", t2)
    return t2


def read_calls(text):
    text = repair_json(text)
    try:
        d = json.loads(text)
        if isinstance(d, dict) and isinstance(d.get("calls"), list):
            return [{"tool": str(c.get("tool", "")).strip(), "args": c.get("args") if isinstance(c.get("args"), dict) else {}}
                    for c in d["calls"] if isinstance(c, dict)], True
    except Exception:
        pass
    return parse_calls(text)


def canon(calls):
    return sorted(json.dumps({"t": c["tool"], "a": {k: str(v).strip().lower() for k, v in sorted((c.get("args") or {}).items())
                                                    if k in ID_FIELDS or k in ("amount_value", "method", "status", "dispute_state", "start_date", "end_date", "carrier", "type")}},
                             sort_keys=True) for c in calls)


def solve(model, task, layers, votes=0):
    """Runs the stack. With layer 3 on, a message is split into its chores and each is solved on its own."""
    parts = segments(task["request"], task["account"]) if "3" in layers else [task["request"]]
    calls, info = [], None
    for p in parts:
        c, i = solve_one(model, task["account"], p, layers, votes, task["request"])
        calls += [x for x in c if x not in calls]
        if info is None:
            info = i
        else:
            for k in ("cost", "secs", "model_calls"):
                info[k] += i[k]
            info["rejections"] += i["rejections"]; info["parsed"] = info["parsed"] and i["parsed"]
            info["raw"] += " || " + i["raw"][:400]
            info["trace"] += i["trace"]
            if "agree" in i:
                info["agree"] = min(info.get("agree", votes), i["agree"])
    info["segments"] = len(parts)
    return calls, info


def solve_one(model, acct, request, layers, votes=0, whole=None):
    """One chore through the stack. Returns (calls, info). `whole` is the full message the chore came from:
    the rule check looks there for amounts and numbers the merchant stated."""
    whole = whole or request
    info = {"cost": 0.0, "secs": 0.0, "model_calls": 0, "rejections": [], "tools_shown": None, "parsed": True, "raw": ""}
    tools = list(SPECS)
    if "3" in layers:
        tools = retrieve_one(request)
    shown_acct = ground(acct, request) if "2" in layers else acct
    schema = None
    if "1" in layers:
        schema, tools = build_schema(tools, shown_acct if "2" in layers else None)
    info["tools_shown"] = len(tools)
    # a step-by-step record of what each layer did, for the results page
    tr = {"seg": request, "tools": list(tools) if "3" in layers else len(tools),
          "records": {c: [len(shown_acct[c]), len(acct[c])] for c in COLLECTIONS + ("products",)},
          "first": [], "rejected": [], "second": None, "final": []}
    info["trace"] = [tr]
    prompt = (PROMPT.replace(OUT_OLD, OUT_NEW) if "1" in layers else PROMPT)
    prompt = prompt.replace("{tools}", json.dumps([SPECS[n]["raw"] for n in tools], indent=1)) \
                   .replace("{account}", json.dumps(shown_acct, indent=1)).replace("{request}", request)
    msgs = [{"role": "user", "content": prompt}]

    def ask(temp=0):
        text, cost, secs = chat(model, msgs, schema, temp)
        info["cost"] += cost; info["secs"] += secs; info["model_calls"] += 1
        return text

    if not tools:
        return [], info
    vmap = {}
    if schema is not None:
        for v in schema["properties"]["calls"]["items"].get("anyOf", []):
            vmap[v["properties"]["tool"]["enum"][0]] = v["properties"]["args"]

    def problems(calls, parsed):
        out = []
        if "1" in layers and not parsed:
            out.append((None, "the reply was not a valid JSON object of the required shape"))
        for c in calls:
            why = template_errors(c, vmap) if "1" in layers else None
            if not why and "4" in layers:
                why = guard(c, acct, whole)
            if why:
                out.append((c, why))
        return out

    def tidy(calls):
        return drop_redundant_lookups(calls, acct) if "4" in layers else calls

    text = ask()
    info["raw"] = text[:1200]
    calls, info["parsed"] = read_calls(text) if "1" in layers else parse_calls(text)
    tr["first"] = [dict(c) for c in calls]
    calls = tidy(calls)
    bad = problems(calls, info["parsed"])
    if bad:
        info["rejections"] = [w for _, w in bad]
        tr["rejected"] = [[c, w] for c, w in bad]
        msgs += [{"role": "assistant", "content": text},
                 {"role": "user", "content": "Your reply was rejected:\n" +
                  "\n".join("- %s%s" % (json.dumps(c) + ": " if c else "", w) for c, w in bad) +
                  '\nReply again with the corrected JSON object. If the request cannot be done, reply {"calls": []}.'}]
        text2 = ask()
        calls2, parsed2 = read_calls(text2)
        tr["second"] = [dict(c) for c in calls2]
        calls2 = tidy(calls2)
        again = problems(calls2, True)
        tr["rejected2"] = [[c, w] for c, w in again if c is not None]
        still = {id(c) for c, _ in again if c is not None}
        calls = [c for c in calls2 if id(c) not in still]            # anything still invalid is not executed
        info["raw2"] = text2[:600]
    if votes:
        msgs_first = msgs[:1]
        agree = 0
        for _ in range(votes):
            t, cost, secs = chat(model, msgs_first, schema, 0.8)
            info["cost"] += cost; info["secs"] += secs; info["model_calls"] += 1
            other, _ = read_calls(t)
            other = tidy(other)
            still = {id(c) for c, _ in problems(other, True) if c is not None}
            other = [c for c in other if id(c) not in still]
            agree += canon(other) == canon(calls)
        info["agree"] = agree
    tr["final"] = calls
    return calls, info


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--layers", default="1234")
    ap.add_argument("--big", default="", help="model to escalate to when layer 5 is on")
    ap.add_argument("--votes", type=int, default=2, help="extra samples for the layer 5 agreement check")
    ap.add_argument("--dev", action="store_true", help="practice set only: every third chore")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--budget", type=int, default=150)
    ap.add_argument("--tag", default="")
    ap.add_argument("--tasks", default="tasks.json", help="which chore file to run")
    a = ap.parse_args(argv)

    global EMBED_LOCAL
    local = a.model.startswith("ollama:") or a.model.startswith("hf:")
    paid = a.model.startswith("or:") or a.big.startswith("or:")
    if local:
        EMBED_LOCAL = a.model.startswith("ollama:")
        a.workers, a.budget = 1, 10 ** 6
    if paid:
        used = key_usage()
        if used >= SPEND_CAP_USD:
            raise SystemExit("Spend cap reached: key usage $%.4f >= cap $%.2f" % (used, SPEND_CAP_USD))
    tasks = json.load(open(os.path.join(HERE, a.tasks)))
    if a.dev:
        tasks = tasks[::3]
    if a.limit:
        tasks = tasks[::max(1, len(tasks) // a.limit)][:a.limit]
    name = re.sub(r"[^A-Za-z0-9.]+", "-", a.model) + "__L" + a.layers + (a.tag and "-" + a.tag) + ("-dev" if a.dev else "")
    out = os.path.join(HERE, "results", name + ".jsonl")
    done = {}
    if os.path.exists(out):
        for line in open(out):
            r = json.loads(line); done[r["id"]] = r
    lock = threading.Lock(); start = time.time(); errs = [0]; stop = threading.Event()
    if "3" in a.layers:
        embed([tool_doc(n) for n in SPECS] + [x for t in tasks for sg in segments(t["request"], t["account"]) for x in [sg] + sentences(sg)])

    def work(t):
        if t["id"] in done or stop.is_set() or time.time() - start > a.budget:
            return
        try:
            esc = False
            calls, info = solve(a.model, t, a.layers, a.votes if "5" in a.layers else 0)
            if "5" in a.layers:
                unsure = bool(info["rejections"]) or info.get("agree", a.votes) < a.votes or not info["parsed"]
                if unsure and a.big:
                    esc = True
                    calls, big = solve(a.big, t, a.layers.replace("5", ""))
                    info["cost_small"] = info["cost"]; info["cost"] += big["cost"]; info["secs"] += big["secs"]
                    info["model_calls"] += big["model_calls"]; info["big_raw"] = big["raw"][:400]
        except Blocked as e:
            stop.set()
            print("  the gateway has paused this key for too many errors; stopping.", flush=True)
            return
        except Exception as e:
            with lock:
                errs[0] += 1
                print("  error on %s: %s" % (t["id"], str(e)[:220]), flush=True)
                if errs[0] >= 3:
                    stop.set()
            return
        ok, why = grade(calls, t["expected"])
        rec = {"id": t["id"], "category": t["category"], "pass": ok, "reason": why, "calls": calls, "escalated": esc}
        rec.update(info)
        with lock:
            errs[0] = 0; done[t["id"]] = rec
            with open(out, "a") as f:
                f.write(json.dumps(rec) + "\n")
            if local:
                print("  %s %-4s %3d/%d done, %d right so far (%.0fs)" % (t["id"], "ok" if ok else "FAIL", len(done), len(tasks), sum(r["pass"] for r in done.values()), info["secs"]), flush=True)

    with cf.ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(work, tasks))
    rs = [done[t["id"]] for t in tasks if t["id"] in done]
    if stop.is_set():
        print("STOPPED after repeated errors.")
    n = max(1, len(rs))
    print("%s | answered %d/%d | passed %d (%.0f%%) | cost $%.4f | escalated %d%s" % (
        name, len(rs), len(tasks), sum(r["pass"] for r in rs), 100.0 * sum(r["pass"] for r in rs) / n,
        sum(r["cost"] for r in rs), sum(r.get("escalated", False) for r in rs),
        " | key usage now $%.4f" % key_usage() if paid else (" | ran on this computer, no cost" if local else " | through the gateway")))
    return 0 if len(rs) == len(tasks) else 2


if __name__ == "__main__":
    sys.exit(main())
