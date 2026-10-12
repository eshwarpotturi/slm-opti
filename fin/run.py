"""Runs a small model on financial-report questions, alone or with the improvement layers.

  python3 fin/run.py bedrock:google.gemma-3-4b-it --layers 0      the model alone, thinking step by step
  python3 fin/run.py bedrock:google.gemma-3-4b-it --layers 1      1 = calculator: the model also writes its arithmetic as a formula; code redoes it and that result is used
  python3 fin/run.py bedrock:google.gemma-3-4b-it --layers 12     2 = source check: every figure in the formula must be in the report; one retry
  python3 fin/run.py bedrock:google.gemma-3-4b-it --layers 123    3 = worked examples: the four most similar solved questions from the training split
  python3 fin/run.py bedrock:google.gemma-3-4b-it --layers 1234   4 = vote: five attempts, the most common result wins
  python3 fin/run.py bedrock:google.gemma-3-4b-it --layers 126    6 = formula rules (fin/rules.py): checked against the question; one retry

Results go to fin/results/<model>__L<layers>-<tag>.jsonl. Resumable.
"""
import argparse
import ast
import collections
import concurrent.futures as cf
import json
import math
import operator
import os
import re
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import stack  # noqa: E402  (model access only)
import rules  # noqa: E402  (layer 6: checks a formula against the question)

stack.PRICES.setdefault("us.anthropic.claude-haiku-4-5-20251001-v1:0", (1.0, 5.0))


# ---------- the report extract, as the model sees it ----------
def render(t):
    rows = [" | ".join(c.strip() for c in r) for r in t["table"]]
    return "\n".join(["\n".join(t["before"]), "", "TABLE:", "\n".join(rows), "", "\n".join(t["after"])]).strip()


# ---------- scoring ----------
def num(x):
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().lower().replace(",", "").replace("$", "").replace("%", "")
    m = re.fullmatch(r"\(?\s*(-?\d+(?:\.\d+)?)\s*\)?\s*(million|billion|thousand)?s?", s)
    if not m:
        return None
    v = float(m.group(1))
    return -abs(v) if str(x).strip().startswith("(") else v


def close(a, b):
    return abs(a - b) <= max(0.01 * abs(b), 1e-9)


def correct(pred, gold):
    """Right if it equals the benchmark's answer to within 1%. A share may be given as 0.07 or as 7 (percent)."""
    if isinstance(gold, str):
        return str(pred).strip().lower() == gold.strip().lower()
    p = num(pred)
    if p is None:
        return False
    return any(close(v, gold) for v in (p, p * 100, p / 100))


# ---------- layer 1: the calculator ----------
OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Pow: operator.pow,
       ast.Gt: operator.gt, ast.Lt: operator.lt, ast.GtE: operator.ge, ast.LtE: operator.le}
FUNCS = {"abs": abs, "max": max, "min": min, "sum": lambda *a: sum(a[0]) if len(a) == 1 and isinstance(a[0], list) else sum(a),
         "average": lambda *a: sum(a) / len(a), "avg": lambda *a: sum(a) / len(a), "round": round}


def clean_formula(f):
    f = str(f).replace("×", "*").replace("÷", "/").replace("−", "-").replace("–", "-").replace("^", "**")
    f = re.sub(r"(?<=\d),(?=\d{3}\b)", "", f)
    f = re.sub(r"[$€£]", "", f)
    f = re.sub(r"(?i)\b(millions?|billions?|thousands?|usd|dollars?|shares?|bps|basis points)\b", "", f)
    f = re.sub(r"(\d)\s*%", r"\1/100", f)
    if "=" in f and not re.search(r"[<>]=", f):
        f = f.split("=")[0]
    return f.strip()


def calc(formula):
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            return float(n.value)
        if isinstance(n, ast.BinOp) and type(n.op) in OPS:
            return OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            return -ev(n.operand) if isinstance(n.op, ast.USub) else ev(n.operand)
        if isinstance(n, ast.Compare) and len(n.ops) == 1 and type(n.ops[0]) in OPS:
            return OPS[type(n.ops[0])](ev(n.left), ev(n.comparators[0]))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in FUNCS and not n.keywords:
            return FUNCS[n.func.id](*[ev(a) for a in n.args])
        if isinstance(n, (ast.List, ast.Tuple)):
            return [ev(e) for e in n.elts]
        raise ValueError("only numbers and + - * / ( ) > < are allowed")
    v = ev(ast.parse(clean_formula(formula), mode="eval"))
    return ("yes" if v else "no") if isinstance(v, bool) else v


def literals(formula):
    out = []
    for n in ast.walk(ast.parse(clean_formula(formula), mode="eval")):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            out.append(float(n.value))
    return out


# ---------- layer 2: the source check ----------
def doc_numbers(text):
    found = set()
    for m in re.finditer(r"\d[\d,]*\.?\d*", text):
        s = m.group(0).replace(",", "").rstrip(".")
        try:
            found.add(round(float(s), 6))
        except ValueError:
            pass
    return found


PLAIN = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 100.0, 1000.0, 1000000.0, 365.0}


def unsupported(formula, doc, question):
    have = doc_numbers(doc + " " + question)
    return [v for v in literals(formula) if round(abs(v), 6) not in have and abs(v) not in PLAIN]


# ---------- prompts ----------
ALONE = ("You are a financial analyst. Read the extract from a company's annual report and answer the question.\n"
         "Work it out step by step, then finish with one line in exactly this form:\nANSWER: <a single number, or yes or no>\n"
         "Do not put units or words on that line. Give a percentage as the percent number, for example 7.52 for 7.52%. Keep at least two decimal places.")
REFS = ("You are a financial analyst. Read the extract from a company's annual report and answer the question.\n"
        "Every table cell has a reference such as R2C3 (row 2, column 3). First write which cells you need, one per line, "
        "as: R2C3 = <row name>, <column name>. Check that each one is the row and the year the question asks about.\n"
        "Then finish with one line:\nFORMULA: <arithmetic using the cell references, for example (R1C1 - R1C2) / R1C2>\n"
        "Use a plain number only for a figure that appears in the text rather than the table. "
        "If the question asks yes or no, finish with ANSWER: yes or ANSWER: no instead.")
EXTRA = ("\nAfter the ANSWER line add one more line with the arithmetic behind it, numbers and signs only, "
        "using figures copied exactly from the extract:\nFORMULA: <for example (5829 - 5735) / 5735>\n"
        "A calculator will redo that arithmetic, so the formula must be complete. For a yes or no question leave the FORMULA line out.")
TUNED = "Read the extract from a company's annual report and write the formula that answers the question."
TUNED_FIG = ("Read the extract from a company's annual report. First list the figures the question needs, one per line, "
             "each with what it is and its year. Then write FORMULA: and the arithmetic that answers the question.")
VOTES = 5


def facts(t):
    """Layer 5: every table cell written out with its row and column names, so no figure has to be found by position."""
    head, out = [c.strip() for c in t["table"][0]], []
    for r in t["table"][1:]:
        cells = ["%s: %s" % (h or "value", c.strip()) for h, c in zip(head[1:], r[1:]) if c.strip()]
        out.append("- %s -> %s" % (r[0].strip(), " ; ".join(cells)))
    return "\n".join(["\n".join(t["before"]), "", "TABLE (%s), one row per line:" % (head[0] or "figures"), "\n".join(out), "", "\n".join(t["after"])]).strip()


LAYERS = ""


def labelled(t):
    """Layer 7: every table cell carries a reference such as R2C3, written as the row name, the column name and the value."""
    head = [c.strip() for c in t["table"][0]]
    out = []
    for i, r in enumerate(t["table"][1:], start=1):
        cells = ["[R%dC%d] %s = %s" % (i, j, head[j] if j < len(head) and head[j] else "column %d" % j, c.strip())
                 for j, c in enumerate(r) if j and c.strip()]
        out.append("%s: %s" % (r[0].strip(), "; ".join(cells)))
    return "\n".join(["\n".join(t["before"]), "", "TABLE, one row per line, each cell with its reference:", "\n".join(out), "", "\n".join(t["after"])]).strip()


def resolve(t, formula):
    """Replaces cell references (R2C3) in a formula with the numbers in those cells."""
    def cell(m):
        i, j = int(m.group(1)), int(m.group(2))
        try:
            v = t["table"][i][j]
        except IndexError:
            raise ValueError("no cell R%dC%d" % (i, j))
        s2 = v.replace("$", "").replace(",", "").replace("%", "").strip()
        neg = s2.startswith("(") and s2.endswith(")")
        s2 = s2.strip("()").split()[0] if s2.strip("()").split() else ""
        float(s2)
        return ("(-%s)" % s2) if neg else s2
    return re.sub(r"\[?R(\d+)C(\d+)\]?", cell, formula)


def ask(t):
    if "7" in LAYERS:
        return "EXTRACT:\n%s\n\nQUESTION: %s" % (labelled(t), t["question"])
    if "5" in LAYERS:
        return "QUESTION: %s\n\nEXTRACT:\n%s\n\nQUESTION (again): %s" % (t["question"], facts(t), t["question"])
    return "EXTRACT:\n%s\n\nQUESTION: %s" % (render(t), t["question"])


def read_alone(text):
    m = re.findall(r"ANSWER\s*:\s*\**\s*([^\n*]+)", text or "", flags=re.I)
    if m:
        a = m[-1].strip().rstrip(".")
        return a.lower() if a.lower() in ("yes", "no") else (num(a) if num(a) is not None else a)
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text or "")
    return num(nums[-1]) if nums else None


def read_formula(text):
    m = re.findall(r"FORMULA\s*:\s*\**\s*`?([^\n`]+)", text or "", flags=re.I)
    return m[-1].strip().strip("*").strip() if m else None


# ---------- layer 3: worked examples from the training split ----------
_EX = {}


def words(q):
    return [w for w in re.findall(r"[a-z]+", q.lower()) if len(w) > 2]


def examples(question, k=4):
    """The k solved training questions most alike in wording (TF-IDF cosine). None of them is in any test file."""
    if not _EX:
        pool = [x for x in json.load(open(os.path.join(HERE, "data", "train.json"))) if x.get("formula") and len(x["facts"]) <= 4]
        pool = [x for x in pool if _safe(x)]
        df = collections.Counter(w for x in pool for w in set(words(x["question"])))
        idf = {w: math.log(len(pool) / c) for w, c in df.items()}
        vecs = []
        for x in pool:
            v = collections.Counter(words(x["question"]))
            v = {w: c * idf[w] for w, c in v.items()}
            n = math.sqrt(sum(a * a for a in v.values())) or 1.0
            vecs.append({w: a / n for w, a in v.items()})
        _EX.update(pool=pool, idf=idf, vecs=vecs)
    q = collections.Counter(words(question))
    q = {w: c * _EX["idf"].get(w, 0.0) for w, c in q.items()}
    n = math.sqrt(sum(a * a for a in q.values())) or 1.0
    scored = sorted(((sum(a / n * v.get(w, 0.0) for w, a in q.items()), i) for i, v in enumerate(_EX["vecs"])), reverse=True)
    out, seen = [], set()
    for _, i in scored:
        x = _EX["pool"][i]
        if x["question"] in seen:
            continue
        seen.add(x["question"])
        out.append(x)
        if len(out) == k:
            break
    return out


def _safe(x):
    try:
        return correct(calc(x["formula"]), x["answer"])
    except Exception:
        return False


def show_number(x):
    """The answer as a person would write it: a share or a rate of change as a percent, anything else as it is."""
    v = x["answer"]
    return "%.2f" % (v * 100) if (abs(v) < 2 and "/" in x["formula"]) else "%g" % round(v, 2)


def example_text(question, with_formula=True):
    lines = ["Here are similar questions from other reports, already solved. They show how to set the answer out; their figures are not yours."]
    for x in examples(question):
        last = "ANSWER: %s" % x["answer"] if isinstance(x["answer"], str) else ("ANSWER: %s\nFORMULA: %s" % (show_number(x), x["formula"]) if with_formula else "Working: %s\nANSWER: %s" % (x["formula"], show_number(x)))
        lines.append("Q: %s\nFigures: %s\n%s" % (x["question"], " ".join(x["facts"])[:400], last))
    return "\n\n".join(lines)


# ---------- one question ----------
def attempt(model, t, layers, temperature, tuned):
    """One pass: returns (answer, formula, cost, notes). The calculator's result replaces the model's own arithmetic."""
    doc, cost, notes = render(t), 0.0, []
    use_calc = tuned or "1" in layers
    use_calc = use_calc or "7" in layers
    head = TUNED if tuned else (REFS if "7" in layers else ALONE + (EXTRA if use_calc else "")) + ("\n\n" + example_text(t["question"], use_calc) if "3" in layers else "")
    msgs = [{"role": "user", "content": head + "\n\n" + ask(t)}]
    answer = formula = None
    for turn in range(2 if (("2" in layers or "6" in layers) and use_calc) else 1):
        text, c, _ = stack.chat(model, msgs, temperature=temperature, max_tokens=600)
        cost += c
        if not use_calc:
            return read_alone(text), None, cost, notes
        said = None if tuned else read_alone(text)
        if tuned and re.search(r"(FORMULA|ANSWER)\s*:", text or ""):
            yn = re.findall(r"ANSWER\s*:\s*(yes|no)\b", text, flags=re.I)
            formula = yn[-1] if yn else read_formula(text)
        else:
            formula = text.strip().splitlines()[0].replace("FORMULA:", "").strip() if (tuned and text.strip()) else read_formula(text)
        if tuned and formula and formula.lower() in ("yes", "no"):
            return formula.lower(), None, cost, notes
        if said in ("yes", "no"):
            return said, None, cost, notes
        if answer is None:
            answer = said
        problem = None
        if not formula:
            problem = "Your reply had no FORMULA line. Give the FORMULA line for your answer"
        else:
            try:
                if "7" in layers:
                    formula = resolve(t, formula)
                value = calc(formula)
                if isinstance(value, str):
                    raise ValueError("comparison")
                bad = unsupported(formula, doc, t["question"]) if "2" in layers else []
                if bad:
                    problem = ("These figures in your formula are not printed in the extract: %s. Use only figures copied from the extract "
                               "and put every step of the arithmetic in the formula. Reply again with the ANSWER and FORMULA lines"
                               % ", ".join("%g" % b for b in bad))
                else:
                    answer = value
                    broken = rules.check(t, formula) if "6" in layers else []
                    if broken:
                        problem = " ".join(broken) + (" Reply with the corrected formula only" if tuned else " Reply again with the ANSWER and FORMULA lines")
            except ZeroDivisionError:
                problem = "Your formula divides by zero. Reply again with the ANSWER and FORMULA lines"
            except Exception:
                problem = "The calculator could not read your formula. Use only numbers and + - * / and brackets. Reply again with the ANSWER and FORMULA lines"
        if not problem:
            break
        notes.append(problem.split(".")[0].split(":")[0])
        if tuned and "6" not in layers:
            break
        msgs += [{"role": "assistant", "content": text[-1500:]}, {"role": "user", "content": problem + "."}]
    return answer, formula, cost, notes


def solve(model, t, layers, tuned=False):
    if not layers.strip("05") and not tuned:
        text, cost, _ = stack.chat(model, [{"role": "user", "content": ALONE + "\n\n" + ask(t)}], temperature=0, max_tokens=500)
        return {"answer": read_alone(text), "cost": cost, "raw": text[-600:]}
    if "4" not in layers:
        a, f, cost, notes = attempt(model, t, layers, 0, tuned)
        return {"answer": a, "formula": f, "cost": cost, "notes": notes}
    tries, cost = [], 0.0
    for i in range(VOTES):
        a, f, c, notes = attempt(model, t, layers, 0 if i == 0 else 0.7, tuned)
        cost += c
        tries.append((a, f, notes))

    def key(a):
        return "%.3g" % a if isinstance(a, float) else a
    keyed = collections.Counter(key(a) for a, _, _ in tries if a is not None)
    if not keyed:
        return {"answer": None, "cost": cost, "votes": [], "notes": tries[0][2]}
    top = keyed.most_common(1)[0][0]
    a, f, notes = next(x for x in tries if x[0] is not None and key(x[0]) == top)
    return {"answer": a, "formula": f, "cost": cost, "agree": keyed[top], "votes": list(keyed.items()), "notes": notes,
            "first": tries[0][0], "first_formula": tries[0][1]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("--layers", default="0"); ap.add_argument("--tasks", default="data/test.json")
    ap.add_argument("--tag", default="t"); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--budget", type=float, default=170); ap.add_argument("--tuned", action="store_true")
    a = ap.parse_args(argv)
    global LAYERS
    LAYERS = a.layers
    tasks = json.load(open(os.path.join(HERE, a.tasks)))
    if a.limit:
        tasks = tasks[:a.limit]
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    out = os.path.join(HERE, "results", "%s__L%s-%s.jsonl" % (re.sub(r"[^A-Za-z0-9.-]+", "-", a.model.split(":", 1)[1]), a.layers, a.tag))
    done = {json.loads(l)["id"] for l in open(out)} if os.path.exists(out) else set()
    todo = [t for t in tasks if t["id"] not in done]
    lock, t0, errs, n_done = threading.Lock(), time.time(), [0], [0]
    print("%s  starting: %d to do, %d already done" % (time.strftime("%H:%M:%S"), len(todo), len(done)), flush=True)

    def work(t):
        if time.time() - t0 > a.budget or errs[0] >= 3:
            return
        t1 = time.time()
        try:
            r = solve(a.model, t, a.layers, a.tuned)
            errs[0] = 0
        except Exception as e:
            errs[0] += 1
            print(time.strftime("%H:%M:%S"), "error", t["id"], repr(e)[:160], flush=True)
            return
        r.update(id=t["id"], gold=t["answer"], right=correct(r["answer"], t["answer"]))
        r["secs"] = round(time.time() - t1, 1)
        if "first" in r:
            r["first_right"] = correct(r["first"], t["answer"])
        with lock:
            open(out, "a").write(json.dumps(r) + "\n")
            n_done[0] += 1
            print("%s  %d/%d  %s  %s  %.0fs" % (time.strftime("%H:%M:%S"), len(done) + n_done[0], len(tasks), t["id"],
                                                "right" if r["right"] else "wrong", r["secs"]), flush=True)

    with cf.ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(work, todo))
    rs = [json.loads(l) for l in open(out)] if os.path.exists(out) else []
    print("%s  L%s  %d/%d done  right %d (%.0f%%)  cost $%.4f" % (a.model, a.layers, len(rs), len(tasks), sum(r["right"] for r in rs),
          100.0 * sum(r["right"] for r in rs) / max(1, len(rs)), sum(r["cost"] for r in rs)))


if __name__ == "__main__":
    main()
