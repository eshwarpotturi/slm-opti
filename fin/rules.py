"""Rules that check a formula against the question and the report table, before the answer is accepted.

Each rule returns a short message for the model when the formula breaks it, or None. They only look at the
question's wording, the table's headers and the formula, so they apply to any question of the same kind.
"""
import re

import run

SHARE = re.compile(r"\b(percent(age)?|portion|proportion|ratio|fraction)\b|%")
CHANGE = re.compile(r"\b(change|increase|decrease|difference|grow(th|n)?|grew|decline|declined|rise|fall|fell|drop)\b")
PCT_CHANGE = re.compile(r"\b(percent(age)?\s+(change|increase|decrease|growth|decline)|growth\s+rate|rate\s+of\s+(change|growth))\b")
YEAR = re.compile(r"\b(19[89]\d|20[0-3]\d)\b")


def _num(s):
    s = s.replace(",", "").replace("$", "").replace("%", "").strip()
    m = re.fullmatch(r"\(?-?\d+(\.\d+)?\)?", s.split(" ")[0]) if s else None
    if not m:
        return None
    v = float(s.split(" ")[0].strip("()"))
    return -v if s.startswith("(") else v


def cell_years(t):
    """For each number in the table, the years named in its column header or its row label."""
    out = {}
    table = t["table"]
    if not table:
        return out
    head = table[0]
    for row in table[1:]:
        for j, cell in enumerate(row):
            if j == 0:
                continue
            v = _num(cell)
            if v is None:
                continue
            years = set(YEAR.findall(head[j] if j < len(head) else "")) | set(YEAR.findall(row[0]))
            out.setdefault(round(abs(v), 6), set()).update(years)
    return out


def table_has_years(t):
    return bool(t["table"]) and any(YEAR.search(h) for h in t["table"][0][1:])


def degenerate(formula):
    """A figure subtracted from or divided by itself, e.g. (843 - 843) / 843 or 254000 / 254000."""
    return bool(re.search(r"(?<![\d.])([\d.]+)\s*[-/]\s*\1(?![\d.])", run.clean_formula(formula)))


def check(t, formula):
    """Problems with this formula for this question, as messages the model can act on."""
    q = t["question"].lower()
    f = run.clean_formula(formula)
    problems = []
    if degenerate(formula):
        problems.append("Your formula uses the same figure twice, so the answer is 0 or 1. Use two different figures from the table.")
    if SHARE.search(q) and "/" not in f and ">" not in f and "<" not in f:
        problems.append("The question asks for a percentage, portion or ratio, so the formula must divide one figure by another.")
    if CHANGE.search(q) and not SHARE.search(q) and "-" not in f and "/" not in f:
        problems.append("The question asks for a change or difference, so the formula must subtract one figure from another.")
    return problems
