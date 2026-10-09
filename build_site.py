"""Builds index.html, the front page that joins both studies. Run: python3 build_site.py
Financial-report figures are read from fin/results. Payment-request figures are the measured ones recorded in README.md."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
F = os.path.join(HERE, "fin")
L = lambda f: {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(F, "results", f + ".jsonl"))}
g, m, h = L("google.gemma-3-4b-it__L0-t"), L("mistral.ministral-3-3b-instruct__L0-t"), L("us.anthropic.claude-haiku-4-5-20251001-v1-0__L0-t")
tu, x, x0 = L("gemma-3-4b-tuned__L2-t"), L("crosscheck-tuned-t"), L("crosscheck-t")
N = len(g)
pct = lambda r, ids=None: round(100.0 * sum(r[i]["right"] for i in (ids or r)) / len(ids or r))
cost = lambda r: 1000.0 * sum(v["cost"] for v in r.values()) / len(r)
AG = [i for i in x if x[i]["route"] == "agreed"]
ES = [i for i in x if x[i]["route"] != "agreed"]
AG0 = [i for i in x0 if x0[i]["route"] == "agreed"]
dev = lambda f: pct(L(f))

# payment requests, 150 unseen requests (README.md)
P = dict(alone=55, alone_w=48, l4=79, l4_w=3, l5=89, l5_w=3, l5_cost=1.12, big=93, big_w=7, big_cost=11.41, alone_cost=0.14,
         ml=78, ml_w=21, mlr=89, mlr_w=4, ft=80, ft_w=20, ftr=89, ftr_w=5, ft_base=51)
LAD = [("Gemma 3 4B", [55, 55, 62, 65, 79], [48, 46, 27, 24, 3]), ("Llama 3.1 8B", [45, 59, 65, 71, 85], [13, 23, 14, 21, 3]),
       ("Ministral 3B", [60, 59, 71, 68, 83], [25, 25, 23, 16, 3])]

INK, MUTE, LINE, GREY, GREEN, SLATE, RED = "var(--ink)", "var(--mute)", "var(--line)", "var(--grey)", "var(--green)", "var(--slate)", "var(--red)"


def svg(w, hgt, body, label):
    return '<svg viewBox="0 0 %d %d" role="img" aria-label="%s" preserveAspectRatio="xMidYMid meet">%s</svg>' % (w, hgt, label, body)


def t(x_, y, s, size=13, fill=INK, anchor="start", weight=400, fam="s"):
    return '<text x="%.1f" y="%.1f" font-size="%d" fill="%s" text-anchor="%s" font-weight="%d" class="%s">%s</text>' % (x_, y, size, fill, anchor, weight, fam, s)


def dumbbell(rows, label):
    """rows: (name, before, after, large). One line per task: grey dot to green dot, slate tick for the large model."""
    W, H, x0_, x1 = 560, 60 + 84 * len(rows), 180, 520
    sx = lambda v: x0_ + (x1 - x0_) * (v - 40) / 60.0
    b = ""
    for v in (40, 60, 80, 100):
        b += '<line x1="%.1f" y1="34" x2="%.1f" y2="%d" stroke="%s"/>' % (sx(v), sx(v), H - 14, LINE) + t(sx(v), 24, "%d%%" % v, 12, MUTE, "middle")
    for k, (name, a, c, big) in enumerate(rows):
        y = 82 + 84 * k
        b += t(0, y - 4, name[0], 15, INK, weight=600) + t(0, y + 14, name[1], 12, MUTE)
        b += '<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="%s" stroke-width="6" stroke-linecap="round" opacity=".28"/>' % (sx(a), y, sx(c), y, GREEN)
        b += '<circle cx="%.1f" cy="%d" r="9" fill="%s"/><circle cx="%.1f" cy="%d" r="11" fill="%s"/>' % (sx(a), y, GREY, sx(c), y, GREEN)
        b += '<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="%s" stroke-width="3"/>' % (sx(big), y - 16, sx(big), y + 16, SLATE)
        b += t(sx(a), y + 30, "%d%%" % a, 13, MUTE, "middle") + t(sx(c), y - 18, "%d%%" % c, 16, GREEN, "middle", 700)
        b += t(sx(big) + (8 if big > c else 8), y + 30, "large %d%%" % big, 12, SLATE, "start" if big >= c else "start")
    return svg(W, H, b, label)


def scatter(points, xmax, label, xticks):
    """points: (name, cost, pct, colour, dx, dy, anchor). Cost on a log axis."""
    import math
    W, H, l, r, top, bot = 400, 300, 46, 384, 16, 248
    lo, hi = math.log10(xticks[0]), math.log10(xmax)
    sx = lambda c: l + (r - l) * (math.log10(c) - lo) / (hi - lo)
    sy = lambda p: bot - (bot - top) * (p - 40) / 60.0
    b = ""
    for v in (40, 60, 80, 100):
        b += '<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="%s"/>' % (l, sy(v), r, sy(v), LINE) + t(l - 8, sy(v) + 4, "%d%%" % v, 12, MUTE, "end")
    for v in xticks:
        b += t(sx(v), bot + 20, "$%g" % v, 12, MUTE, "middle")
    b += t((l + r) / 2, H - 8, "cost per 1,000 requests (log scale)", 12, MUTE, "middle")
    for name, c, p, col, dx, dy, an in points:
        b += '<circle cx="%.1f" cy="%.1f" r="%d" fill="%s"/>' % (sx(c), sy(p), 10 if col == GREEN else 8, col)
        b += t(sx(c) + dx, sy(p) + dy, name, 13, col if col != GREY else MUTE, an, 600 if col == GREEN else 400)
    return svg(W, H, b, label)


def flow():
    W, H = 720, 250
    a, e = len(AG), len(ES)
    ha, he = 150.0 * a / N, 150.0 * e / N
    b = '<rect x="0" y="50" width="150" height="150" rx="4" fill="%s" opacity=".5"/>' % GREY
    b += t(75, 118, str(N), 30, INK, "middle", 700, "d") + t(75, 140, "questions", 13, INK, "middle")
    b += '<rect x="270" y="50" width="120" height="150" rx="4" fill="none" stroke="%s" stroke-width="1.5"/>' % INK
    b += t(330, 112, "two small", 13, INK, "middle") + t(330, 130, "models answer", 13, INK, "middle") + t(330, 152, "same number?", 13, INK, "middle", 700)
    b += '<path d="M150 125 H270" stroke="%s" stroke-width="2" fill="none"/>' % MUTE
    b += '<path d="M390 %.1f C450 %.1f 450 %.1f 510 %.1f L510 %.1f C450 %.1f 450 %.1f 390 %.1f Z" fill="%s" opacity=".35"/>' % (50, 50, 20, 20, 20 + ha, 20 + ha, 50 + ha, 50 + ha, GREEN)
    b += '<path d="M390 %.1f C450 %.1f 450 %.1f 510 %.1f L510 %.1f C450 %.1f 450 %.1f 390 %.1f Z" fill="%s" opacity=".3"/>' % (50 + ha, 50 + ha, 160, 160, 160 + he, 160 + he, 200, 200, SLATE)
    b += '<rect x="510" y="20" width="10" height="%.1f" fill="%s"/><rect x="510" y="160" width="10" height="%.1f" fill="%s"/>' % (ha, GREEN, he, SLATE)
    b += t(532, 44, "%d agree" % a, 20, GREEN, weight=700, fam="d") + t(532, 64, "answer used as it is", 13) + t(532, 83, "%d%% of these are correct" % pct(x, AG), 13, GREEN, weight=700)
    b += t(532, 186, "%d differ" % e, 20, SLATE, weight=700, fam="d") + t(532, 206, "sent to a larger model", 13) + t(532, 225, "%d%% of these are correct" % pct(x, ES), 13, SLATE)
    return svg(W, H, b, "Flow of the %d questions through the cross-check" % N)


def ladder():
    W, H, l, r, top, bot = 720, 320, 44, 540, 24, 262
    names = ["Model alone", "+ reply check", "+ record lookup", "+ split, search", "+ rule check"]
    sx = lambda k: l + (r - l) * k / 4.0
    sy = lambda p: bot - (bot - top) * (p - 40) / 60.0
    b = ""
    for v in (40, 60, 80, 100):
        b += '<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="%s"/>' % (l, sy(v), r, sy(v), LINE) + t(l - 8, sy(v) + 4, "%d%%" % v, 12, MUTE, "end")
    for k, n in enumerate(names):
        b += t(sx(k), bot + 22, n, 12, MUTE, "middle" if 0 < k < 4 else ("start" if k == 0 else "end"))
    cols = [GREEN, SLATE, MUTE]
    for (name, pts, _), col in zip(LAD, cols):
        d = " ".join("%s%.1f %.1f" % ("M" if k == 0 else "L", sx(k), sy(p)) for k, p in enumerate(pts))
        b += '<path d="%s" fill="none" stroke="%s" stroke-width="%s"/>' % (d, col, "3" if col == GREEN else "2")
        for k, p in enumerate(pts):
            b += '<circle cx="%.1f" cy="%.1f" r="4" fill="%s"/>' % (sx(k), sy(p), col)
    ends = sorted(((sy(pts[-1]), name, pts[-1], col) for (name, pts, _), col in zip(LAD, cols)))
    y = 0
    for yy, name, p, col in ends:
        y = max(yy, y + 20)
        b += t(r + 14, y + 4, "%s  %d%%" % (name, p), 13, col, weight=600)
    return svg(W, H, b, "Share of payment requests handled correctly as each layer is added")


def wrong_bars():
    rows = [("Small model alone", P["alone_w"], GREY), ("With the layers of code", P["l4_w"], GREEN), ("Large model alone", P["big_w"], SLATE)]
    W, H = 380, 168
    b = ""
    for k, (n, v, col) in enumerate(rows):
        y = 10 + 52 * k
        b += t(0, y + 12, n, 13) + '<rect x="0" y="%d" width="%.1f" height="16" rx="2" fill="%s"/>' % (y + 20, max(4, 6.6 * v), col)
        b += t(max(4, 6.6 * v) + 8, y + 33, str(v), 15, col if col != GREY else INK, weight=700)
    return svg(W, H, b, "Wrong money-moving actions out of 150 requests")


def dots():
    base = dev("google.gemma-3-4b-it__L0-d")
    rows = [("Calculator redoes the arithmetic", dev("google.gemma-3-4b-it__L1-d3")), ("Calculator + check figures against the page", dev("google.gemma-3-4b-it__L12-d3")),
            ("Those two + similar solved examples", dev("google.gemma-3-4b-it__L123-d3")), ("Similar solved examples only", dev("google.gemma-3-4b-it__L3-d4")),
            ("Five attempts, majority answer", dev("google.gemma-3-4b-it__L4-d4")), ("Table rewritten row by row", dev("google.gemma-3-4b-it__L5-d5"))]
    W, H, l, r = 720, 46 + 34 * len(rows), 300, 700
    sx = lambda v: l + (r - l) * (v - 45) / 30.0
    b = '<line x1="%.1f" y1="26" x2="%.1f" y2="%d" stroke="%s" stroke-width="2" stroke-dasharray="4 4"/>' % (sx(base), sx(base), H - 6, INK)
    b += t(sx(base), 16, "model alone %d%%" % base, 12, INK, "middle", 600)
    for k, (n, v) in enumerate(rows):
        y = 48 + 34 * k
        b += t(0, y + 4, n, 13) + '<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="%s" stroke-width="3"/>' % (sx(base), y, sx(v), y, RED)
        b += '<circle cx="%.1f" cy="%d" r="7" fill="%s"/>' % (sx(v), y, RED if v < base else GREY) + t(sx(v) - 14 if v < base else sx(v) + 14, y + 4, "%d%%" % v, 13, MUTE, "end" if v < base else "start")
    return svg(W, H, b, "Six code-only techniques tried on the financial questions, none above the model alone")


def grouped():
    rows = [("Classical ML, no language model", P["ml"], P["mlr"]), ("Fine-tuned small model", P["ft"], P["ftr"]), ("Untuned small model", P["alone"], P["l4"])]
    tail = ["with the rule check", "with the rule check", "with all four layers of code"]
    W, H = 720, 200
    sx = lambda v: 250 + 3.2 * v
    b = ""
    for k, (n, a, c) in enumerate(rows):
        y = 34 + 54 * k
        b += t(0, y + 22, n, 13)
        b += '<rect x="250" y="%d" width="%.1f" height="14" rx="2" fill="%s"/>' % (y, 3.2 * a, GREY) + t(sx(a) + 6, y + 12, "%d%% alone" % a, 12, MUTE)
        b += '<rect x="250" y="%d" width="%.1f" height="14" rx="2" fill="%s"/>' % (y + 18, 3.2 * c, GREEN) + t(sx(c) + 6, y + 30, "%d%% %s" % (c, tail[k]), 12, GREEN, weight=600)
    return svg(W, H, b, "Payment requests: classical machine learning against small language models, with and without the rule check")


def mini_tiles(vals, hi):
    """A 10 by 10 grid of squares, hi of them filled."""
    b = ""
    for i in range(100):
        b += '<rect x="%d" y="%d" width="9" height="9" rx="1.5" fill="%s"/>' % (11 * (i % 20), 11 * (i // 20), vals if i < hi else LINE)
    return svg(218, 53, b, "%d of 100" % hi)


d = dict(
    N=N, fin_a=pct(g), fin_b=pct(x), fin_big=pct(h), fin_ratio="%.1f" % (cost(h) / cost(x)), fin_cost="%.2f" % cost(x), big_cost="%.2f" % cost(h),
    na=len(AG), pa=pct(x, AG), pa0=pct(x0, AG0), ne=len(ES), pt=pct(tu), pm=pct(m), px0=pct(x0),
    hero=dumbbell([(("Financial reports", "work out a figure from a report page"), pct(g), pct(x), pct(h)),
                   (("Payment requests", "choose and fill in the right action"), P["alone"], P["l5"], P["big"])],
                  "Small model alone against our setup and a large model, on two tasks"),
    flow=flow(),
    sc_fin=scatter([("small model alone", cost(g), pct(g), GREY, 12, 4, "start"), ("our setup", cost(x), pct(x), GREEN, -14, -10, "end"),
                    ("large model", cost(h), pct(h), SLATE, 0, 22, "middle")], 4, "Financial reports: accuracy against cost", [0.05, 0.2, 1, 4]),
    sc_pay=scatter([("small model alone", P["alone_cost"], P["alone"], GREY, 12, 4, "start"), ("our setup", P["l5_cost"], P["l5"], GREEN, -14, -8, "end"),
                    ("large model", P["big_cost"], P["big"], SLATE, -14, 4, "end")], 16, "Payment requests: accuracy against cost", [0.1, 1, 10]),
    ladder=ladder(), wrong=wrong_bars(), dots=dots(), grouped=grouped(),
    t_ag=mini_tiles(GREEN, pct(x, AG)), t_al=mini_tiles(GREY, pct(g)),
    **{"p_" + k: v for k, v in P.items()})
page = open(os.path.join(HERE, "site_template.html")).read()
for k, v in d.items():
    page = page.replace("{{%s}}" % k, str(v))
assert "{{" not in page, page[page.index("{{"):][:60]
import re
keep = lambda txt, tag, drop: re.sub(r"<!--%s-->.*?<!--/%s-->" % (drop, drop), "", txt, flags=re.S).replace("<!--%s-->" % tag, "").replace("<!--/%s-->" % tag, "")
open(os.path.join(HERE, "full.html"), "w").write(keep(page, "ML", "PUB"))      # the complete account, kept for reference
page = keep(page, "PUB", "ML")
open(os.path.join(HERE, "index.html"), "w").write(page)
print("index.html", len(page) // 1024, "KB", {k: v for k, v in d.items() if isinstance(v, (int, str)) and len(str(v)) < 8})
