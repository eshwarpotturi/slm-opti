"""Prints a score table for every finished run in results/."""
import glob, json, os
from collections import Counter, defaultdict
here = os.path.dirname(os.path.abspath(__file__))
cats = ["1_direct", "2_lookup", "3_list_create", "4_multi_step", "5_should_not_act"]
print("%-44s %5s %6s | %s | %6s" % ("run", "n", "pass%", "  ".join(c[:6].ljust(6) for c in cats), "sec"))
for f in sorted(glob.glob(os.path.join(here, "results", "*.jsonl"))):
    rs = list({json.loads(l)["id"]: json.loads(l) for l in open(f)}.values())
    by = defaultdict(list)
    for r in rs: by[r["category"]].append(r["pass"])
    cell = "  ".join(("%3.0f%%" % (100.0 * sum(by[c]) / len(by[c])) if by[c] else "  - ").ljust(6) for c in cats)
    print("%-44s %5d %5.0f%% | %s | %6.2f" % (os.path.basename(f)[:-6], len(rs), 100.0 * sum(r["pass"] for r in rs) / len(rs), cell, sum(r["secs"] for r in rs) / len(rs)))
    print("    failures:", dict(Counter(r["reason"] for r in rs if not r["pass"])), "| unparsed:", sum(not r["parsed"] for r in rs))
