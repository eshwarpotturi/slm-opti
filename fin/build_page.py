"""Builds fin/index.html from fin/results. Run: python3 fin/build_page.py"""
import html
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
L = lambda f: {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(HERE, "results", f + ".jsonl"))}
T = {t["id"]: t for t in json.load(open(os.path.join(HERE, "data", "test.json")))}
g, m = L("google.gemma-3-4b-it__L0-t"), L("mistral.ministral-3-3b-instruct__L0-t")
h, x = L("us.anthropic.claude-haiku-4-5-20251001-v1-0__L0-t"), L("crosscheck-t")
N = len(T)


def pct(r, ids=None):
    ids = list(ids or r)
    return round(100.0 * sum(r[i]["right"] for i in ids) / len(ids))


def per1000(r):
    return 1000.0 * sum(v["cost"] for v in r.values()) / len(r)


agreed = [i for i, v in x.items() if v["route"] == "agreed"]
esc = [i for i in x if i not in set(agreed)]
tuned = None
if os.path.exists(os.path.join(HERE, "results", "gemma-3-4b-tuned__L2-t.jsonl")):
    tuned = L("gemma-3-4b-tuned__L2-t")

bars = [("Gemma 3 4B alone", pct(g), per1000(g), "s"), ("Ministral 3B alone", pct(m), per1000(m), "s")]
if tuned:
    bars.append(("Gemma 3 4B, fine-tuned", pct(tuned), 0, "u"))
bars += [("Both small models, cross-checked, larger model on disagreements", pct(x), per1000(x), "u"),
         ("Larger model alone (Claude Haiku 4.5)", pct(h), per1000(h), "b")]


def fmt(v):
    if v is None:
        return "no answer"
    return v if isinstance(v, str) else ("%.4g" % v)


rows = []
for i, t in T.items():
    rows.append("<tr class='%s'><td>%s</td><td class='n'>%s</td><td class='n %s'>%s</td><td class='n %s'>%s</td><td>%s</td><td class='n %s'>%s</td></tr>" % (
        x[i]["route"], html.escape(t["question"]), fmt(t["answer"]), "ok" if g[i]["right"] else "no", fmt(g[i]["answer"]),
        "ok" if m[i]["right"] else "no", fmt(m[i]["answer"]), "matched" if x[i]["route"] == "agreed" else "differed, sent up",
        "ok" if x[i]["right"] else "no", fmt(x[i]["answer"])))

tried = [("The model alone, thinking step by step", "google.gemma-3-4b-it__L0-d"),
         ("Calculator: code redoes the model's arithmetic", "google.gemma-3-4b-it__L1-d3"),
         ("Calculator and a check that every figure is printed in the report", "google.gemma-3-4b-it__L12-d3"),
         ("Those two, plus four similar solved questions as examples", "google.gemma-3-4b-it__L123-d3"),
         ("Examples only", "google.gemma-3-4b-it__L3-d4"),
         ("Five attempts, most common answer wins", "google.gemma-3-4b-it__L4-d4"),
         ("Table rewritten one labelled row per line", "google.gemma-3-4b-it__L5-d5")]
tried_rows = "".join("<tr><td>%s</td><td class='n'>%d%%</td></tr>" % (a, pct(L(f))) for a, f in tried)

page = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Small models on financial reports</title><style>
:root{--bg:#fbfaf7;--ink:#1c1b19;--mute:#6b675f;--line:#dedad2;--s:#b9b3a6;--u:#1f6f5c;--b:#3a4a6b;--ok:#1f6f5c;--no:#a8432f;--card:#fff}
@media (prefers-color-scheme:dark){:root{--bg:#161614;--ink:#ece9e2;--mute:#a09a8e;--line:#34322d;--s:#5d594f;--u:#4fb89c;--b:#8fa3d0;--ok:#4fb89c;--no:#e08670;--card:#1e1d1b}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 Georgia,serif}
main{max-width:860px;margin:0 auto;padding:40px 16px 80px}h1{font-size:2rem;line-height:1.2;margin:0 0 12px}h2{font-size:1.25rem;margin:44px 0 10px}
p{margin:0 0 12px}.lead{font-size:1.1rem}.mute{color:var(--mute);font-size:.9rem}
.bar{margin:14px 0}.bar .lab{display:flex;justify-content:space-between;gap:12px;font-family:system-ui,sans-serif;font-size:.92rem}
.track{background:var(--line);height:22px;border-radius:3px;overflow:hidden;margin-top:4px}.fill{height:100%%}
.s{background:var(--s)}.u{background:var(--u)}.b{background:var(--b)}
table{border-collapse:collapse;width:100%%;font-family:system-ui,sans-serif;font-size:.86rem}th,td{border-bottom:1px solid var(--line);padding:7px 8px;text-align:left;vertical-align:top}
.n{text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums}.ok{color:var(--ok)}.no{color:var(--no)}
.wrap{overflow-x:auto;background:var(--card);border:1px solid var(--line);border-radius:4px}.tall{max-height:520px;overflow:auto}
.flow{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;font-family:system-ui,sans-serif;font-size:.9rem}
.flow div{background:var(--card);border:1px solid var(--line);border-radius:4px;padding:12px}.flow b{display:block;font-size:1.5rem}
button{font:inherit;font-family:system-ui,sans-serif;font-size:.85rem;padding:5px 10px;margin:0 6px 8px 0;border:1px solid var(--line);background:var(--card);color:var(--ink);border-radius:3px;cursor:pointer}
</style></head><body><main>
<h1>Two small models that check each other reach %(px)d%%, against %(ph)d%% for a larger model at nearly twice the cost</h1>
<p class="lead">The task is the kind an analyst does all day: read a page of a company's annual report and work out a figure from it, such as a margin, a growth rate or a share of a total. The test is %(N)d questions from FinQA, a public benchmark written by finance experts, restricted to banks, insurers and payment companies.</p>

<h2>Share of questions answered correctly</h2>
%(bars)s
<p class="mute">An answer counts as correct when it is within 1%% of the benchmark's answer. A share may be given as 0.07 or as 7 percent. Cost is at list prices per 1,000 questions.</p>

<h2>How the cross-check works</h2>
<div class="flow">
<div><b>%(N)d</b>questions go to two small models, each working alone</div>
<div><b>%(na)d</b>get the same answer from both. Those answers are used as they are, and %(pa)d%% of them are correct</div>
<div><b>%(ne)d</b>get different answers. Those go to the larger model, which gets %(pe)d%% of them correct</div>
<div><b>%(px)d%%</b>correct overall, at %(ratio)d%% of the cost of sending everything to the larger model</div>
</div>
<p style="margin-top:14px">A small model alone is right a little over half the time, and gives no sign of which half. Agreement between two different small models is the sign: when they agree they are right about as often as the larger model is on those same questions (%(ha)d%%).</p>

<h2>What did not help</h2>
<p>Before the test, each of these was tried on Gemma 3 4B with 100 questions from the benchmark's separate development split. None beat the model working alone, so none was carried to the test. With 100 questions, differences of a few points are within chance.</p>
<div class="wrap"><table><tr><th>Setup</th><th class="n">Correct</th></tr>%(tried)s</table></div>
<p style="margin-top:12px">The reason is visible in the wrong answers. The arithmetic is rarely the problem. The model picks the wrong row or year, or misreads what the question asks for, and a calculator cannot repair that.</p>

<h2>By difficulty</h2>
<div class="wrap"><table><tr><th>Arithmetic steps in the answer</th><th class="n">Questions</th><th class="n">Gemma alone</th><th class="n">Ministral alone</th><th class="n">Cross-check</th><th class="n">Larger model</th></tr>%(steps)s</table></div>

<h2>Every question</h2>
<p><button onclick="f('')">All %(N)d</button><button onclick="f('agreed')">Small models matched</button><button onclick="f('escalated')">Sent to the larger model</button></p>
<div class="wrap tall"><table id="q"><tr><th>Question</th><th class="n">Benchmark</th><th class="n">Gemma</th><th class="n">Ministral</th><th>Route</th><th class="n">Final</th></tr>%(rows)s</table></div>

<h2>Limits</h2>
<p>Half the gain comes from the larger model, not from the small ones. The benchmark hands the model the right page; finding the page in a full report is a separate problem not tested here. FinQA's own answers contain some errors, which caps every score. All runs are single runs.</p>
<p class="mute">Data: FinQA (Chen et al., 2021), MIT licence. Code and every recorded answer are in the repository.</p>
</main><script>function f(c){document.querySelectorAll('#q tr').forEach(function(r,i){if(i)r.style.display=(!c||r.className==c)?'':'none'})}</script></body></html>"""

bar_html = "".join("<div class='bar'><div class='lab'><span>%s</span><span>%d%%%s</span></div><div class='track'><div class='fill %s' style='width:%d%%'></div></div></div>" % (
    a, p, (" · $%.2f" % c) if c else "", k, p) for a, p, c, k in bars)
steps = ""
for k, lab in [(1, "One"), (2, "Two"), (3, "Three or more")]:
    ids = [i for i, t in T.items() if min(t["steps"], 3) == k]
    steps += "<tr><td>%s</td><td class='n'>%d</td><td class='n'>%d%%</td><td class='n'>%d%%</td><td class='n'>%d%%</td><td class='n'>%d%%</td></tr>" % (
        lab, len(ids), pct(g, ids), pct(m, ids), pct(x, ids), pct(h, ids))
out = page % dict(N=N, bars=bar_html, na=len(agreed), pa=pct(x, agreed), ne=len(esc), pe=pct(x, esc), px=pct(x), ph=pct(h), ha=pct(h, agreed),
                  ratio=round(100 * per1000(x) / per1000(h)), tried=tried_rows, steps=steps, rows="".join(rows))
open(os.path.join(HERE, "index.html"), "w").write(out)
print("written", len(out) // 1024, "KB", bars)
