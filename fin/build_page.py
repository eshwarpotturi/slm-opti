"""Builds fin/index.html from fin/results. Run: python3 fin/build_page.py"""
import html
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
L = lambda f: {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(HERE, "results", f + ".jsonl"))}
T = {t["id"]: t for t in json.load(open(os.path.join(HERE, "data", "test.json")))}
g, m = L("google.gemma-3-4b-it__L0-t"), L("mistral.ministral-3-3b-instruct__L0-t")
h, x0 = L("us.anthropic.claude-haiku-4-5-20251001-v1-0__L0-t"), L("crosscheck-t")
tu, x = L("gemma-3-4b-tuned__L2-t"), L("crosscheck-tuned-t")
N = len(T)
E = html.escape


def pct(r, ids=None):
    ids = list(ids or r)
    return round(100.0 * sum(r[i]["right"] for i in ids) / len(ids))


def per1000(r):
    return 1000.0 * sum(v["cost"] for v in r.values()) / len(r)


def fmt(v, gold=None):
    if v is None:
        return "no answer"
    if isinstance(v, str):
        return v
    if isinstance(gold, float) and abs(gold) < 1 and abs(v) < 1:
        return "%.1f%%" % (v * 100)
    return "%.4g" % v


agreed = [i for i, v in x.items() if v["route"] == "agreed"]
esc = [i for i in x if x[i]["route"] != "agreed"]

# one real example: the small model alone was wrong, the pair agreed and was right, short question and table
cands = [i for i in agreed if x[i]["right"] and not g[i]["right"] and g[i]["answer"] is not None and isinstance(T[i]["answer"], float)
         and 3 <= len(T[i]["table"]) <= 6 and len(T[i]["table"][0]) <= 4 and len(T[i]["question"]) < 110 and T[i]["steps"] == 2]
ex = T[sorted(cands, key=lambda i: len(json.dumps(T[i]["table"])))[0]]
ex_table = "".join("<tr>" + "".join("<%s>%s</%s>" % (("th" if r == 0 or c == 0 else "td class='n'"), E(cell), "th" if r == 0 or c == 0 else "td")
                                    for c, cell in enumerate(row)) + "</tr>" for r, row in enumerate(ex["table"]))

rows = []
for i, t in T.items():
    rows.append("<tr class='%s'><td>%s</td><td class='n'>%s</td><td class='n %s'>%s</td><td class='n %s'>%s</td><td>%s</td><td class='n %s'>%s</td></tr>" % (
        x[i]["route"], E(t["question"]), fmt(t["answer"]), "ok" if tu[i]["right"] else "no", fmt(tu[i]["answer"]),
        "ok" if m[i]["right"] else "no", fmt(m[i]["answer"]), "agreed" if x[i]["route"] == "agreed" else "sent up",
        "ok" if x[i]["right"] else "no", fmt(x[i]["answer"])))

tried = [("Model alone, thinking step by step", "google.gemma-3-4b-it__L0-d"),
         ("Calculator: code redoes the model's arithmetic", "google.gemma-3-4b-it__L1-d3"),
         ("Calculator plus a check that every figure is printed in the report", "google.gemma-3-4b-it__L12-d3"),
         ("Those two, plus four similar solved questions as examples", "google.gemma-3-4b-it__L123-d3"),
         ("Examples only", "google.gemma-3-4b-it__L3-d4"),
         ("Five attempts, most common answer wins", "google.gemma-3-4b-it__L4-d4"),
         ("Table rewritten one labelled row per line", "google.gemma-3-4b-it__L5-d5")]
tried_rows = "".join("<tr><td>%s</td><td class='n'>%d%%</td></tr>" % (a, pct(L(f))) for a, f in tried)

bars = [("One small model alone (Gemma 3 4B)", pct(g), "$%.2f" % per1000(g), "s"),
        ("Another small model alone (Ministral 3B)", pct(m), "$%.2f" % per1000(m), "s"),
        ("Gemma after fine-tuning, alone", pct(tu), "", "s"),
        ("The pair without fine-tuning, plus escalation", pct(x0), "$%.2f" % per1000(x0), "u2"),
        ("Our setup: fine-tuned pair, plus escalation", pct(x), "$%.2f*" % per1000(x), "u"),
        ("Larger model on everything (Claude Haiku 4.5)", pct(h), "$%.2f" % per1000(h), "b")]
bar_html = "".join("<div class='bar'><div class='lab'><span>%s</span><span><b>%d%%</b>%s</span></div><div class='track'><div class='fill %s' style='width:%d%%'></div></div></div>" % (
    a, p, (" &nbsp;" + c) if c else "", k, p) for a, p, c, k in bars)

steps = ""
for k, lab in [(1, "One"), (2, "Two"), (3, "Three or more")]:
    ids = [i for i, t in T.items() if min(t["steps"], 3) == k]
    steps += "<tr><td>%s</td><td class='n'>%d</td><td class='n'>%d%%</td><td class='n'>%d%%</td><td class='n'>%d%%</td><td class='n'>%d%%</td></tr>" % (
        lab, len(ids), pct(g, ids), pct(tu, ids), pct(x, ids), pct(h, ids))

page = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Small models on financial reports</title><style>
:root{--bg:#f7f6f2;--ink:#1b1a18;--mute:#6a665e;--line:#dcd8cf;--s:#b5afa2;--u:#17705b;--u2:#7fb5a6;--b:#3b4d75;--no:#a8432f;--card:#fff;--soft:#eaf3f0}
@media (prefers-color-scheme:dark){:root{--bg:#151514;--ink:#ece9e2;--mute:#a19b8f;--line:#35332e;--s:#625e54;--u:#4fb89c;--u2:#2f6f60;--b:#8fa3d0;--no:#e08670;--card:#1e1d1b;--soft:#1b2a26}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:900px;margin:0 auto;padding:36px 16px 80px}
h1{font:600 2.1rem/1.15 Georgia,serif;margin:0 0 10px;text-wrap:balance}h2{font:600 1.3rem/1.2 Georgia,serif;margin:52px 0 6px}
.sub{color:var(--mute);margin:0 0 16px}.lead{font-size:1.1rem;margin:0 0 26px;max-width:62ch}
.tiles{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.tile{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:18px}
.tile .k{font-size:.82rem;color:var(--mute);text-transform:uppercase;letter-spacing:.04em}.tile .v{font:600 3rem/1 Georgia,serif;margin:8px 0 6px}
.tile .d{font-size:.92rem}.tile.win{border-color:var(--u);background:var(--soft)}.tile.win .v{color:var(--u)}.grey{color:var(--s)}.blue{color:var(--b)}
.flow{display:grid;grid-template-columns:1fr auto 1fr auto 1fr;gap:10px;align-items:stretch}
.step{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:16px}.step .n1{font:600 1.9rem/1 Georgia,serif}
.step small{display:block;color:var(--mute);margin-top:6px}.arr{align-self:center;color:var(--mute);font-size:1.4rem}
.split{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:10px}.good{border-color:var(--u);background:var(--soft)}
.ex{display:grid;grid-template-columns:1.1fr 1fr;gap:14px}.ex .q{font:600 1.05rem/1.35 Georgia,serif;margin:0 0 10px}
.ans{display:flex;justify-content:space-between;gap:10px;padding:10px 0;border-bottom:1px solid var(--line)}.ans:last-child{border:0}
.ans b{font-variant-numeric:tabular-nums}.bad{color:var(--no)}.okc{color:var(--u)}
.bar{margin:12px 0}.bar .lab{display:flex;justify-content:space-between;gap:12px;font-size:.93rem}.track{background:var(--line);height:20px;border-radius:4px;overflow:hidden;margin-top:4px}
.fill{height:100%%}.fill.s{background:var(--s)}.fill.u{background:var(--u)}.fill.u2{background:var(--u2)}.fill.b{background:var(--b)}
table{border-collapse:collapse;width:100%%;font-size:.86rem}th,td{border-bottom:1px solid var(--line);padding:7px 8px;text-align:left;vertical-align:top}
.n{text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums}.ok{color:var(--u)}.no{color:var(--no)}
.wrap{overflow-x:auto;background:var(--card);border:1px solid var(--line);border-radius:8px}.tall{max-height:480px;overflow:auto}
details{border-top:1px solid var(--line);padding:14px 0}details:last-of-type{border-bottom:1px solid var(--line)}summary{cursor:pointer;font-weight:600}
details p{color:var(--mute);margin:10px 0}.note{color:var(--mute);font-size:.86rem;margin-top:10px}
button{font:inherit;font-size:.85rem;padding:5px 10px;margin:8px 6px 8px 0;border:1px solid var(--line);background:var(--card);color:var(--ink);border-radius:4px;cursor:pointer}
@media (max-width:700px){.tiles,.ex,.split{grid-template-columns:1fr}.flow{grid-template-columns:1fr}.arr{transform:rotate(90deg);justify-self:center}h1{font-size:1.6rem}}
</style></head><body><main>
<h1>Small models, made as accurate as a large one on financial reports, at about half the cost</h1>
<p class="lead">The job: read a page of a company's annual report and work out a number from it, such as a growth rate or a share of a total. Tested on %(N)d questions written by finance experts about banks, insurers and payment companies.</p>

<div class="tiles">
<div class="tile"><div class="k">Small model alone</div><div class="v grey">%(pg)d%%</div><div class="d">correct. Cheap, but wrong nearly half the time, with no warning.</div></div>
<div class="tile win"><div class="k">Our setup</div><div class="v">%(px)d%%</div><div class="d">correct. Two small models check each other; only the doubtful questions go to a larger model.</div></div>
<div class="tile"><div class="k">Larger model on everything</div><div class="v blue">%(ph)d%%</div><div class="d">correct, at about %(mult).1f times the cost of our setup.</div></div>
</div>

<h2>How it works</h2>
<p class="sub">Two different small models answer every question separately. If they agree, trust it. If they differ, ask a larger model.</p>
<div class="flow">
<div class="step"><div class="n1">%(N)d</div>questions<small>each answered by two small models, one of them fine-tuned by us</small></div>
<div class="arr">&rarr;</div>
<div class="step"><div class="n1">Same answer?</div><small>code compares the two numbers</small></div>
<div class="arr">&rarr;</div>
<div><div class="step good"><div class="n1">%(na)d agree</div>answer used as it is<small><b>%(pa)d%% of these are correct</b></small></div>
<div class="step" style="margin-top:10px"><div class="n1">%(ne)d differ</div>sent to the larger model<small>%(pe)d%% of these are correct</small></div></div>
</div>

<h2>One real question from the test</h2>
<div class="ex">
<div class="wrap"><table>%(ex_table)s</table></div>
<div><p class="q">&ldquo;%(ex_q)s&rdquo;</p>
<div class="ans"><span>Correct answer</span><b>%(ex_gold)s</b></div>
<div class="ans"><span>Small model alone</span><b class="bad">%(ex_g)s &#10007;</b></div>
<div class="ans"><span>Fine-tuned small model</span><b class="okc">%(ex_t)s &#10003;</b></div>
<div class="ans"><span>Second small model</span><b class="okc">%(ex_m)s &#10003;</b></div>
<div class="ans"><span>They agree, so the answer is used</span><b class="okc">no larger model needed</b></div></div>
</div>

<h2>All setups compared</h2>
<p class="sub">Share of the %(N)d questions answered correctly, and cost per 1,000 questions.</p>
%(bars)s
<p class="note">Correct means within 1%% of the benchmark's answer. Costs are list prices. * Does not include the cost of running the fine-tuned model.</p>

<h2>Details</h2>
<details open><summary>What we did to the small model</summary>
<p>Fine-tuning: Gemma 3 4B was trained for 36 minutes on a free GPU, on 1,026 solved questions from the benchmark's training split, to write the formula for the answer. Code then does the arithmetic and checks that every figure in the formula is printed in the report. Alone, the model rose from %(pg)d%% to %(pt)d%%. Paired with a second small model, the answers the two agree on are %(pa)d%% correct, up from %(pa0)d%% for the pair without fine-tuning.</p></details>
<details open><summary>What we tried that did not help</summary>
<p>Each of these was tried on Gemma 3 4B with 100 questions from the benchmark's separate development split. None beat the model alone, so none was used. The mistakes are mostly choosing the wrong row or year, not arithmetic. With 100 questions, a few points either way is within chance.</p>
<div class="wrap"><table><tr><th>Setup</th><th class="n">Correct</th></tr>%(tried)s</table></div></details>
<details open><summary>Results by difficulty</summary>
<div class="wrap" style="margin-top:10px"><table><tr><th>Arithmetic steps needed</th><th class="n">Questions</th><th class="n">Small model alone</th><th class="n">Fine-tuned</th><th class="n">Our setup</th><th class="n">Larger model</th></tr>%(steps)s</table></div></details>
<details id="every"><summary>Every question and every answer</summary>
<button onclick="f('')">All %(N)d</button><button onclick="f('agreed')">Small models agreed</button><button onclick="f('escalated')">Sent to the larger model</button>
<div class="wrap tall"><table id="q"><tr><th>Question</th><th class="n">Correct answer</th><th class="n">Fine-tuned Gemma</th><th class="n">Ministral</th><th>Route</th><th class="n">Final</th></tr>%(rows)s</table></div></details>
<details open><summary>Limits</summary>
<p>About half the questions still go to the larger model. The benchmark supplies the right page of the report; finding that page in a full report is not tested. The benchmark's own answers contain some errors, which caps every score. All runs are single runs. The fine-tuned pair's result combines recorded answers from separate runs of each model.</p></details>
<p class="note">Data: FinQA (Chen et al., 2021), MIT licence. Code and every recorded answer are in the repository.</p>
</main><script>function f(c){document.querySelectorAll('#q tr').forEach(function(r,i){if(i)r.style.display=(!c||r.className==c)?'':'none'})}</script></body></html>"""

i = ex["id"]
out = page % dict(N=N, pg=pct(g), px=pct(x), ph=pct(h), pt=pct(tu), mult=per1000(h) / per1000(x), na=len(agreed), pa=pct(x, agreed),
                  pa0=pct(x0, [k for k in x0 if x0[k]["route"] == "agreed"]), ne=len(esc), pe=pct(x, esc), ex_table=ex_table, ex_q=E(ex["question"]),
                  ex_gold=fmt(ex["answer"], ex["answer"]), ex_g=fmt(g[i]["answer"]), ex_t=fmt(tu[i]["answer"], ex["answer"]),
                  ex_m=fmt(m[i]["answer"]) + "%", bars=bar_html, tried=tried_rows, steps=steps, rows="".join(rows))
open(os.path.join(HERE, "index.html"), "w").write(out)
print("written", len(out) // 1024, "KB | example:", ex["id"], ex["question"], ex["answer"], g[i]["answer"], tu[i]["answer"], m[i]["answer"], ex["table"])
