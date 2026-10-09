"""Parsing and scoring. A chore passes only if the model's tool calls match the
expected calls exactly: same tools, same key values, nothing extra."""
import json
import re


def parse_calls(text):
    """Returns (calls, parsed_ok). No JSON found counts as 'took no action'."""
    if not text:
        return [], False
    t = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    t = re.sub(r"```(?:json)?", "", t)
    cands = []
    i, j = t.find("["), t.rfind("]")
    if i != -1 and j > i:
        cands.append(t[i:j + 1])
    i, j = t.find("{"), t.rfind("}")
    if i != -1 and j > i:
        cands.append("[" + t[i:j + 1] + "]")
    for c in cands:
        try:
            data = json.loads(c)
        except Exception:
            continue
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            continue
        calls = []
        for d in data:
            if not isinstance(d, dict):
                continue
            name = d.get("tool") or d.get("name") or d.get("function") or d.get("tool_name")
            args = d.get("args") or d.get("arguments") or d.get("parameters") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            if isinstance(name, str):
                calls.append({"tool": name.strip(), "args": args if isinstance(args, dict) else {}})
        return calls, True
    return [], False


def _num(v):
    try:
        return float(str(v).replace("$", "").replace(",", "").strip())
    except Exception:
        return None


def _same(got, want):
    if isinstance(want, dict) and "$absent" in want:
        return got in (None, "", [], {})
    if isinstance(want, dict) and "$absent_or" in want:
        return got in (None, "") or _same(got, want["$absent_or"])
    if got is None:
        return False
    if isinstance(want, (int, float)) and not isinstance(want, bool):
        g = _num(got)
        return g is not None and abs(g - float(want)) < 0.005
    return str(got).strip().casefold() == str(want).strip().casefold()


def call_ok(got, want):
    if got["tool"] != want["tool"]:
        return False
    return all(_same(got["args"].get(k), v) for k, v in want["args"].items())


def grade(calls, expected):
    """Returns (passed, reason). Order of calls is not checked."""
    if not expected:
        return (True, "ok") if not calls else (False, "acted_when_it_should_not")
    if not calls:
        return False, "did_nothing"
    left = list(calls)
    tools_want = sorted(c["tool"] for c in expected)
    tools_got = sorted(c["tool"] for c in calls)
    for w in expected:
        hit = next((g for g in left if call_ok(g, w)), None)
        if hit is None:
            if tools_got != tools_want:
                return False, "wrong_tool" if len(tools_got) == len(tools_want) else ("missing_step" if len(tools_got) < len(tools_want) else "extra_step")
            return False, "wrong_details"
        left.remove(hit)
    if left:
        return False, "extra_step"
    return True, "ok"
