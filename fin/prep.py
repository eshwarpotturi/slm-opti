"""Cuts the public FinQA benchmark (Chen et al., 2021, MIT licence) down to its banking, insurance and payments companies.

  python3 fin/prep.py <path to FinQA/dataset>

Writes fin/data/test.json (every such question in FinQA's test split), dev.json and train.json.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SECTOR = set("GS PNC C FIS GPN STT CME BLK ETFC AON V MA MS HIG SPGI TROW FITB CB L RE JKHY MKTX".split())
NAMES = ("jpmorgan", "j.p. morgan", "jp morgan", "paypal")
HELD = {"JPM"}       # kept out of the public files


SIGN = {"add": "+", "subtract": "-", "multiply": "*", "divide": "/", "exp": "**", "greater": ">"}


def infix(prog):
    """FinQA writes divide(subtract(a, b), b). This returns (a - b) / b, or None for its whole-table operations."""
    prog = prog.replace(" ", "")

    def parse(i):
        m = re.match(r"([a-z_]+)\(", prog[i:])
        if m:
            op, i = m.group(1), i + m.end()
            a, i = parse(i)
            b, i = parse(i + 1)
            if op not in SIGN:
                raise ValueError(op)
            return "(%s %s %s)" % (a, SIGN[op], b), i + 1
        m = re.match(r"[^,()]+", prog[i:])
        tok = m.group(0)
        if tok.startswith("const_"):
            tok = tok[6:].replace("m", "-")
        float(tok.rstrip("%"))
        return tok.replace("%", "/100") if tok.endswith("%") else tok, i + m.end()
    try:
        out, _ = parse(0)
        return out[1:-1] if out.startswith("(") else out
    except Exception:
        return None


def slim(x):
    q = x["qa"]
    return {"id": x["id"], "before": x["pre_text"], "table": x["table"], "after": x["post_text"],
            "question": q["question"], "answer": q["exe_ans"], "program": q["program_re"], "steps": len(q["steps"]),
            "formula": infix(q["program_re"]), "facts": list(q["gold_inds"].values())}


src = sys.argv[1]
for split in ("train", "dev", "test"):
    d = json.load(open(os.path.join(src, split + ".json")))
    pub = [slim(x) for x in d if x["id"].split("/")[0] in SECTOR]
    pub = [x for x in pub if not any(w in json.dumps(x).lower() for w in NAMES)]
    json.dump(pub, open(os.path.join(HERE, "data", split + ".json"), "w"))
    held = [slim(x) for x in d if x["id"].split("/")[0] in HELD]
    if split == "test":
        os.makedirs(os.path.join(HERE, "private"), exist_ok=True)
        json.dump(held, open(os.path.join(HERE, "private", "bank.json"), "w"))
    print(split, len(pub), "public,", len(held), "held back")
