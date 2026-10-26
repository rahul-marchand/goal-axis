#!/usr/bin/env python3
"""Build the retargeting-cost pilot report: one standalone page, build-time SVG, no runtime JS.

    python3 figures/src/retarget-pilot/build_fig.py [--data data.json] [--notes notes.json] [--out report.html]

``data.json`` comes from ``scripts/studies/retarget/retarget_report.py --data``;
``notes.json`` holds the prose (keyed by section) so that rewording a caption
never touches drawing code. The page has a short core and a long appendix:

  core      1 the metric for every agent against the share of mazes the goal switches
            2 every agent at every goal: excess loss over its own reference, against n
            3 the metric per goal, with the range the null-arm subtraction leaves
            4 noise: each pipeline stage's spread against the signal
  appendix  one goal's raw learning curves, every agent's learning curves, total cost across the sweep, where the
            bits go, the step early stopping chose, the cost as n grows

A *series* is one agent under one learner: the reference full-weight learner
(``main``) or the learner restricted to the goal vector (``dial only``).

Shares the blog fragments' look (fonts, tokens, frames) but, like every builder
here, none of their code.
"""

import argparse
import json
import math
import pathlib

SRC = pathlib.Path(__file__).resolve().parent

SANS = "'IBM Plex Sans', system-ui, sans-serif"
SERIF = "'Source Serif 4', Georgia, serif"
MONO = "'IBM Plex Mono', ui-monospace, monospace"
INK = "var(--ink)"
SOFT = "var(--ink-soft)"
GRID = "var(--grid)"

NULL_TAU = 10.0
COMPETENT = 1.5
"""Bits per route the null arm may start at for an agent to count as competent.

Subtracting the null arm removes a competence gap only when that gap and the
retargeting are learned independently. At the memoriser's ~100 bits per route
they are learned together, so its difference means nothing. The learner starts
at 0.33 and the goal-vector agent at 1.03, which is why the gate is not at 1:
the goal-vector agent's arms do share a smaller competence gap (its null arm
learns about 0.2 bits per route), and the report says so where it matters.
"""

# (agent, variant) -> name, colour, dash
SERIES = {
    ("bcnv11.s1", "main"): ("Learner (bcnv11)", "var(--learner)", None),
    ("bcgv11.s1", "main"): ("Goal-vector agent", "var(--write)", None),
    ("bcgv11.s1", "dial only"): ("Goal-vector agent, dial-only learner", "var(--write)", "5 3"),
    ("bcov11.n1k.s1", "main"): ("Memoriser", "var(--memoriser)", None),
    ("bcnv11.s2", "main"): ("Learner, seed 2", "var(--learner)", "2 3"),
    ("bcnv11.s3", "main"): ("Learner, seed 3", "var(--learner)", "2 3"),
}
CORE_SERIES = [("bcnv11.s1", "main"), ("bcgv11.s1", "main"), ("bcgv11.s1", "dial only")]
PANEL_TITLE = {
    ("bcnv11.s1", "main"): "Learner, all weights",
    ("bcgv11.s1", "main"): "Goal-vector, all weights",
    ("bcgv11.s1", "dial only"): "Goal-vector, dial only",
}


def tau_colour(tau):
    """Rates below the base rate in the poor goal's hue, above it in the rich one's, null in ink."""
    if tau == NULL_TAU:
        return INK
    shade = min(abs(tau - NULL_TAU) / 9.5, 1.0)
    hue = "--rich" if tau > NULL_TAU else "--poor"
    return f"color-mix(in oklab, var({hue}) {int(45 + 55 * shade)}%, var(--panel))"


def fmt_bits(value):
    if abs(value) >= 10_000:
        return f"{value / 1000:.0f}k"
    if abs(value) >= 1000:
        return f"{value / 1000:.1f}k".replace(".0k", "k")
    return f"{value:g}"


# ----------------------------------------------------------------------
# Drawing primitives
# ----------------------------------------------------------------------


class Chart:
    """A panel with linear or log axes. Ranges of a log axis are in log10."""

    def __init__(self, x, y, w, h, xr, yr, xlog=False, ylog=False):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.xr, self.yr, self.xlog, self.ylog = xr, yr, xlog, ylog

    def px(self, v):
        v = math.log10(v) if self.xlog else v
        return self.x + (v - self.xr[0]) / (self.xr[1] - self.xr[0]) * self.w

    def py(self, v):
        v = math.log10(v) if self.ylog else v
        return self.y + self.h - (v - self.yr[0]) / (self.yr[1] - self.yr[0]) * self.h

    def frame(self, xticks, yticks, xlabel=None, ylabel=None):
        out = []
        for value, _ in yticks:
            py = self.py(value)
            out.append(
                f'<line x1="{self.x}" y1="{py:.1f}" x2="{self.x + self.w}" y2="{py:.1f}" stroke="{GRID}" stroke-width="1"/>'
            )
        out.append(
            f'<line x1="{self.x}" y1="{self.y + self.h}" x2="{self.x + self.w}" y2="{self.y + self.h}" stroke="{SOFT}" stroke-width="1"/>'
        )
        out.append(
            f'<line x1="{self.x}" y1="{self.y}" x2="{self.x}" y2="{self.y + self.h}" stroke="{SOFT}" stroke-width="1"/>'
        )
        for value, text in xticks:
            px = self.px(value)
            out.append(
                f'<line x1="{px:.1f}" y1="{self.y + self.h}" x2="{px:.1f}" y2="{self.y + self.h + 4}" stroke="{SOFT}"/>'
            )
            out.append(text_at(px, self.y + self.h + 16, text, MONO, 10, SOFT, "middle"))
        for value, text in yticks:
            py = self.py(value)
            out.append(f'<line x1="{self.x - 4}" y1="{py:.1f}" x2="{self.x}" y2="{py:.1f}" stroke="{SOFT}"/>')
            out.append(text_at(self.x - 8, py + 3.5, text, MONO, 10, SOFT, "end"))
        if xlabel:
            out.append(label(self.x + self.w / 2, self.y + self.h + 34, xlabel))
        if ylabel:
            out.append(
                f'<text transform="translate({self.x - 48} {self.y + self.h / 2}) rotate(-90)" text-anchor="middle" '
                f'font-size="10" letter-spacing="0.1em" fill="{SOFT}" font-family="{SANS}">{ylabel.upper()}</text>'
            )
        return "".join(out)

    def line(self, points, colour, width=1.8, dash=None, dots=True, r=2.4):
        if not points:
            return ""
        path = " ".join(f"{'M' if i == 0 else 'L'}{self.px(x):.1f},{self.py(y):.1f}" for i, (x, y) in enumerate(points))
        dashed = f' stroke-dasharray="{dash}"' if dash else ""
        out = [f'<path d="{path}" fill="none" stroke="{colour}" stroke-width="{width}"{dashed} stroke-linejoin="round"/>']
        if dots:
            out += [f'<circle cx="{self.px(x):.1f}" cy="{self.py(y):.1f}" r="{r}" fill="{colour}"/>' for x, y in points]
        return "".join(out)


def text_at(x, y, text, family=SANS, size=10.5, fill=SOFT, anchor="start", weight=None):
    bold = f' font-weight="{weight}"' if weight else ""
    return f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-size="{size}" fill="{fill}" font-family="{family}"{bold}>{text}</text>'


def label(x, y, text, anchor="middle"):
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="10" letter-spacing="0.1em" '
        f'fill="{SOFT}" font-family="{SANS}">{text.upper()}</text>'
    )


def legend(x, y, items, step=None):
    out = []
    for colour, text, dash in items:
        dashed = f' stroke-dasharray="{dash}"' if dash else ""
        out.append(f'<line x1="{x}" y1="{y}" x2="{x + 18}" y2="{y}" stroke="{colour}" stroke-width="2"{dashed}/>')
        out.append(text_at(x + 23, y + 3.5, text))
        x += step or (34 + 5.7 * len(text))
    return "".join(out)


def svg(width, height, body, title):
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{title}" '
        f'style="width:100%;max-width:{width}px;height:auto;display:block">{body}</svg>'
    )


def log_ticks(lo, hi, values):
    return [(v, fmt_bits(v)) for v in values if lo <= v <= hi]


# ----------------------------------------------------------------------
# Selecting runs
# ----------------------------------------------------------------------


def runs(rows, agent, variant="main"):
    return sorted((r for r in rows if r["agent"] == agent and r["variant"] == variant), key=lambda r: r["tau"])


def run_at(rows, agent, variant, tau):
    found = [r for r in runs(rows, agent, variant) if r["tau"] == tau]
    return found[0] if found else None


def present(rows, keys):
    return [k for k in keys if runs(rows, *k)]


def competent(rows, agent, variant="main"):
    null = run_at(rows, agent, variant, NULL_TAU)
    return null is not None and null["test_bits"][0] < COMPETENT


# ----------------------------------------------------------------------
# Core figures
# ----------------------------------------------------------------------


def figure_headline(rows, floor_bits):
    """Core 1: the learner's SDL - null against the size of the change, log x, with its seed twins.

    Each point is the per-route null subtraction (the spec's); the bar runs to
    the per-token one. The truth sits somewhere between: an arm's residual error
    differs from the null's by route length and by which objective the routes go
    to, and neither subtraction removes both.
    """
    keys = [
        k for k in present(rows, [("bcnv11.s1", "main"), ("bcnv11.s2", "main"), ("bcnv11.s3", "main")]) if competent(rows, *k)
    ]
    if not keys:
        return ""
    series = {
        k: sorted(
            (r["ceiling_minus_null_raw"], r["sdl_minus_null_raw"], r["sdl_minus_null"], r["tau"])
            for r in runs(rows, *k)
            if r["tau"] != NULL_TAU and r["sdl_minus_null_raw"] is not None
        )
        for k in keys
    }
    values = [v for pts in series.values() for p in pts for v in p[:3] if v is not None and v > 0]
    lo = math.log10(min(values + [floor_bits]) * 0.5)
    hi = math.log10(max(values) * 2)
    width, top, size = 790, 30, 330
    c = Chart(96, top, size * 1.45, size, (lo, hi), (lo, hi), xlog=True, ylog=True)
    ticks = log_ticks(10**lo, 10**hi, [1, 3, 10, 30, 100, 300, 1000, 3000, 10_000, 30_000, 100_000])
    body = [
        c.frame(
            ticks,
            ticks,
            xlabel="size of the behaviour change: n · L(0), less the null arm's (bits)",
            ylabel="retargeting cost (bits)",
        )
    ]
    body.append(c.line([(10**lo, 10**lo), (10**hi, 10**hi)], SOFT, width=1, dash="5 4", dots=False))
    body.append(text_at(c.px(10 ** (hi - 0.1)) - 4, c.py(10 ** (hi - 0.1)) + 4, "pays the whole change", anchor="end"))
    body.append(c.line([(10**lo, floor_bits), (10**hi, floor_bits)], SOFT, width=1, dash="1 3", dots=False))
    body.append(
        text_at(
            c.px(10**hi) - 4, c.py(floor_bits) - 6, f"a perfect dial: name the goal, ≈ {floor_bits:.0f} bits", anchor="end"
        )
    )
    chart_h = size
    for key in keys:
        main = key == ("bcnv11.s1", "main")
        colour = "var(--learner)"
        pts = series[key]
        for x, raw, per_token, _ in pts:
            if per_token is not None and per_token > 0:
                body.append(
                    f'<line x1="{c.px(x):.1f}" y1="{c.py(raw):.1f}" x2="{c.px(x):.1f}" y2="{c.py(per_token):.1f}" stroke="{colour}" stroke-width="{4 if main else 2}" stroke-opacity="0.35"/>'
                )
        body.append(
            c.line(
                [(x, raw) for x, raw, _, _ in pts],
                colour,
                width=2 if main else 1,
                dash=None if main else "2 3",
                dots=True,
                r=3.2 if main else 1.8,
            )
        )
        if main:
            for i, (x, raw, _, tau) in enumerate(pts):
                body.append(
                    text_at(
                        c.px(x) + (6 if i % 2 else -6),
                        c.py(raw) + (-8 if i % 2 else 15),
                        f"τ {tau:g}",
                        MONO,
                        9,
                        SOFT,
                        "start" if i % 2 else "end",
                    )
                )
    body.append(
        legend(
            c.x,
            top + chart_h + 60,
            [("var(--learner)", "learner bcnv11.s1", None), ("var(--learner)", "seeds 2 and 3", "2 3")],
            step=170,
        )
    )
    body.append(
        f'<line x1="{c.x + 360}" y1="{top + chart_h + 54}" x2="{c.x + 360}" y2="{top + chart_h + 66}" stroke="var(--learner)" stroke-width="4" stroke-opacity="0.35"/>'
    )
    body.append(text_at(c.x + 368, top + chart_h + 63.5, "range: null subtracted per route … per token"))
    return svg(width, top + chart_h + 76, "".join(body), "Retargeting cost against size of the behaviour change")


def figure_close_up(rows, tau):
    """Core 2: L(t) at one rate for the core series, each beside its own null arm (dashed grey)."""
    keys = present(rows, CORE_SERIES)
    if not keys:
        return ""
    width, top, chart_w, chart_h = 790, 44, 210, 190
    chosen = [(k, run_at(rows, *k, tau), run_at(rows, *k, NULL_TAU)) for k in keys]
    chosen = [(k, r, null) for k, r, null in chosen if r is not None]
    values = [
        v
        for _, r, null in chosen
        for v in r["test_bits"] + (null["test_bits"] if null else []) + [r.get("oracle_test_bits", r["test_bits"][-1])]
    ]
    n = max(r["n"] for _, r, _ in chosen)
    ylo, yhi = math.log10(min(values) * 0.7), math.log10(max(values) * 1.3)
    body = []
    for i, (key, r, null) in enumerate(chosen):
        x = 70 + i * 250
        c = Chart(x, top, chart_w, chart_h, (math.log10(0.5), math.log10(n * 1.3)), (ylo, yhi), xlog=True, ylog=True)
        _, colour, _ = SERIES[key]
        body.append(text_at(x, top - 14, PANEL_TITLE[key], SANS, 12, INK, weight=600))
        body.append(
            c.frame(
                [(0.5, "0"), (16, "16"), (512, "512"), (n, fmt_bits(n))],
                log_ticks(10**ylo, 10**yhi, [0.1, 0.2, 0.3, 0.5, 1, 2, 5]),
                xlabel="demonstrations seen",
                ylabel="bits per route on T" if i == 0 else None,
            )
        )
        if "oracle_test_bits" in r:
            body.append(
                c.line(
                    [(0.5, r["oracle_test_bits"]), (n * 1.3, r["oracle_test_bits"])],
                    "var(--write)",
                    width=1.2,
                    dash="1 3",
                    dots=False,
                )
            )
            verdict = f"beyond best dial setting: {r['sdl_minus_oracle']:+,.0f} bits"
        elif null:
            body.append(
                c.line(
                    [(max(t, 0.5), v) for t, v in zip(null["bounds"], null["test_bits"])],
                    SOFT,
                    width=1.2,
                    dash="4 3",
                    dots=False,
                )
            )
            verdict = f"over its null arm: {r['sdl_minus_null_raw']:+,.0f} bits"
        else:
            verdict = ""
        body.append(c.line([(max(t, 0.5), v) for t, v in zip(r["bounds"], r["test_bits"])], colour, r=2))
        body.append(text_at(x + chart_w, top + chart_h + 50, verdict, MONO, 10, INK, "end"))
    body.append(
        legend(
            70,
            top + chart_h + 76,
            [
                (SOFT, "learner: its null arm (τ 10)", "4 3"),
                ("var(--write)", "goal-vector agent: its dial set by oracle", "1 3"),
            ],
            step=250,
        )
    )
    return svg(width, top + chart_h + 90, "".join(body), f"Learning curves at tau {tau}")


def nice_ticks(lo, hi, count=4):
    raw = (hi - lo) / count
    step = 10 ** math.floor(math.log10(raw))
    step *= next(m for m in (1, 2, 5, 10) if raw <= m * step)
    first = math.ceil(lo / step) * step
    return [round(first + i * step, 10) for i in range(int((hi - first) / step) + 1)]


def figure_all_agents(rows):
    """Core 2: every agent at every goal, as excess test loss over that agent's own reference.

    The learner's reference is its null arm at the same n; the goal-vector
    agent's is its oracle dial, shifted by what that learner's null arm has
    gained by n (general skill, not retargeting: about 0.2 bits per route for
    all weights by 16k, almost nothing for the dial alone). The memoriser has no
    meaningful reference (it fails the competence gate), so its panel shows raw
    loss on a log axis instead.
    """
    learners = present(rows, [("bcnv11.s1", "main"), ("bcnv11.s2", "main"), ("bcnv11.s3", "main")])
    goalvec = present(rows, [("bcgv11.s1", "main"), ("bcgv11.s1", "dial only")])
    memoriser = ("bcov11.n1k.s1", "main")
    taus = sorted({r["tau"] for k in learners + goalvec for r in runs(rows, *k)} - {NULL_TAU})
    if not taus:
        return ""

    def excess(key, tau):
        r = run_at(rows, *key, tau)
        if r is None:
            return None
        null = run_at(rows, *key, NULL_TAU)
        if null is None:
            return None
        if r.get("oracle_test_bits") is not None:  # the oracle dial, plus the general skill this learner has gained by n
            ref = [r["oracle_test_bits"] + b - null["test_bits"][0] for b in null["test_bits"]]
        else:
            ref = null["test_bits"]
        return [(max(t, 0.5), v - b) for t, v, b in zip(r["bounds"], r["test_bits"], ref)]

    styles = {
        ("bcnv11.s1", "main"): ("var(--learner)", None, 2),
        ("bcnv11.s2", "main"): ("var(--learner)", "1 2.5", 1),
        ("bcnv11.s3", "main"): ("var(--learner)", "1 2.5", 1),
        ("bcgv11.s1", "main"): ("var(--write)", None, 2),
        ("bcgv11.s1", "dial only"): ("var(--write)", "5 3", 2),
    }
    n = max(r["n"] for k in learners + goalvec for r in runs(rows, *k))
    xr = (math.log10(0.5), math.log10(n * 1.3))
    xticks = [(0.5, "0"), (16, "16"), (512, "512"), (n, fmt_bits(n))]
    cols, chart_w, chart_h, gap_x, gap_y, left, top = 4, 140, 118, 52, 84, 62, 40
    body = []
    panels = taus[: 2 * cols - (1 if runs(rows, *memoriser) else 0)]
    for i, tau in enumerate(panels):
        x = left + (i % cols) * (chart_w + gap_x)
        y = top + (i // cols) * (chart_h + gap_y)
        curves = {k: excess(k, tau) for k in learners + goalvec}
        curves = {k: v for k, v in curves.items() if v}
        values = [v for pts in curves.values() for _, v in pts]
        lo, hi = min(min(values), -0.02), max(values)
        pad = (hi - lo) * 0.08
        lo, hi = lo - pad, hi + pad
        c = Chart(x, y, chart_w, chart_h, xr, (lo, hi), xlog=True)
        yt = nice_ticks(lo, hi)
        body.append(
            c.frame(
                xticks,
                [(v, f"{v:g}") for v in yt],
                xlabel="demonstrations" if i // cols == 1 else None,
                ylabel="excess bits / route" if i % cols == 0 else None,
            )
        )
        body.append(c.line([(10 ** xr[0], 0), (10 ** xr[1], 0)], SOFT, width=1, dash="4 3", dots=False))
        flipped = run_at(rows, "bcnv11.s1", "main", tau)
        share = f" · {flipped['flipped_fraction']:.0%} switch" if flipped and flipped.get("flipped_fraction") else ""
        body.append(text_at(x, y - 10, f"τ {tau:g}", MONO, 11.5, INK, weight=600))
        body.append(text_at(x + chart_w, y - 10, share.lstrip(" ·"), SANS, 9.5, SOFT, "end"))
        for key in reversed(list(curves)):
            colour, dash, width = styles[key]
            body.append(c.line(curves[key], colour, width=width, dash=dash, dots=False))
    if runs(rows, *memoriser):
        i = len(panels)
        x = left + (i % cols) * (chart_w + gap_x)
        y = top + (i // cols) * (chart_h + gap_y)
        mem = runs(rows, *memoriser)
        values = [v for r in mem for v in r["test_bits"]]
        ylo, yhi = math.log10(min(values) * 0.8), math.log10(max(values) * 1.25)
        c = Chart(x, y, chart_w, chart_h, xr, (ylo, yhi), xlog=True, ylog=True)
        body.append(c.frame(xticks, log_ticks(10**ylo, 10**yhi, [10, 20, 50, 100]), xlabel="demonstrations", ylabel=None))
        body.append(text_at(x, y - 10, "Memoriser, raw loss", SANS, 11.5, INK, weight=600))
        for r in mem:
            colour = INK if r["tau"] == NULL_TAU else "var(--memoriser)"
            pts = [(max(t, 0.5), v) for t, v in zip(r["bounds"], r["test_bits"])]
            body.append(c.line(pts, colour, width=1.6 if r["tau"] == NULL_TAU else 1, dots=False))
        body.append(text_at(x + chart_w, y + 12, "bits / route, log", MONO, 9, SOFT, "end"))
    ly = top + 2 * chart_h + gap_y + 58
    body.append(
        legend(
            left,
            ly,
            [
                ("var(--learner)", "learner bcnv11", None),
                ("var(--learner)", "seeds 2, 3", "1 2.5"),
                ("var(--write)", "goal-vector, all weights", None),
            ],
            step=None,
        )
    )
    body.append(
        legend(
            left,
            ly + 20,
            [
                ("var(--write)", "goal-vector, dial only", "5 3"),
                (SOFT, "reference level", "4 3"),
                ("var(--memoriser)", "memoriser arms; its null in ink", None),
            ],
            step=None,
        )
    )
    return svg(790, ly + 32, "".join(body), "Every agent at every goal")


def oracle_based(rows, key, tau):
    """An arm's excess over its oracle dial less the null arm's over its own: no composition bias."""
    r, null = run_at(rows, *key, tau), run_at(rows, *key, NULL_TAU)
    if r is None or null is None or r.get("sdl_minus_oracle") is None or null.get("oracle_bits") is None:
        return None
    return r["sdl_minus_oracle"] - (null["mdl"] - null["oracle_bits"])


def figure_overview(rows):
    """Core 1: the metric for every competent agent against the share of mazes the goal change switches.

    Each agent is read against its cleanest reference: the learner against its
    null arm (per route; the bias is small for it), the goal-vector agent
    against its oracle dial. The share of mazes switched is a property of the
    expert alone, so every agent sits on the same x and seeds share a band.
    """
    seeds = present(rows, [("bcnv11.s1", "main"), ("bcnv11.s2", "main"), ("bcnv11.s3", "main")])
    goalvec = present(rows, [("bcgv11.s1", "main"), ("bcgv11.s1", "dial only")])
    if not seeds:
        return ""
    taus = sorted({r["tau"] for r in runs(rows, *seeds[0])} - {NULL_TAU})
    share = {t: run_at(rows, *seeds[0], t)["flipped_fraction"] for t in taus}
    taus = sorted(taus, key=share.get)

    def learner(key):
        return [(share[t], run_at(rows, *key, t)["sdl_minus_null_raw"]) for t in taus if run_at(rows, *key, t)]

    band = [(share[t], [run_at(rows, *k, t)["sdl_minus_null_raw"] for k in seeds if run_at(rows, *k, t)]) for t in taus]
    curves = {k: [(share[t], oracle_based(rows, k, t)) for t in taus] for k in goalvec}
    curves = {k: [(x, v) for x, v in pts if v is not None] for k, pts in curves.items()}
    values = [v for _, vs in band for v in vs] + [v for pts in curves.values() for _, v in pts] + [0]
    lo, hi = min(values), max(values)
    pad = (hi - lo) * 0.06
    xmax = max(share.values()) * 1.08
    width, top, chart_w, chart_h = 790, 24, 600, 330
    c = Chart(96, top, chart_w, chart_h, (0, xmax), (lo - pad, hi + pad))
    xt = [x / 100 for x in nice_ticks(0, xmax * 100, 6)]
    body = [
        c.frame(
            [(x, f"{x:.0%}") for x in xt],
            [(v, fmt_bits(v)) for v in nice_ticks(lo - pad, hi + pad, 6)],
            xlabel="mazes whose choice the new goal switches",
            ylabel="retargeting cost (bits)",
        )
    ]
    body.append(c.line([(0, 0), (xmax, 0)], SOFT, width=1, dash="4 3", dots=False))
    body.append(text_at(c.px(xmax) - 4, c.py(0) - 6, "no cost beyond the reference", anchor="end"))
    upper = [(x, max(vs)) for x, vs in band]
    lower = [(x, min(vs)) for x, vs in band]
    path = " ".join(f"{'M' if i == 0 else 'L'}{c.px(x):.1f},{c.py(v):.1f}" for i, (x, v) in enumerate(upper + lower[::-1]))
    body.append(f'<path d="{path} Z" fill="var(--learner)" fill-opacity="0.16" stroke="none"/>')
    main = learner(("bcnv11.s1", "main"))
    body.append(c.line(main, "var(--learner)", width=2, r=3.2))
    for (x, v), t in zip(main, taus):
        body.append(text_at(c.px(x) - 6, c.py(v) - 9, f"τ {t:g}", MONO, 9, SOFT, "end"))
    for key, pts in curves.items():
        dial = key[1] == "dial only"
        body.append(c.line(pts, "var(--write)", width=2, dash="5 3" if dial else None, dots=False))
        for x, v in pts:
            fill = "var(--panel)" if dial else "var(--write)"
            body.append(
                f'<circle cx="{c.px(x):.1f}" cy="{c.py(v):.1f}" r="3.2" fill="{fill}" stroke="var(--write)" stroke-width="1.5"/>'
            )
    ly = top + chart_h + 58
    body.append(
        legend(
            c.x,
            ly,
            [
                ("var(--learner)", "learner bcnv11, vs its null arm", None),
                ("var(--write)", "goal-vector, all weights, vs its oracle dial", None),
            ],
            step=290,
        )
    )
    body.append(
        f'<rect x="{c.x}" y="{ly + 14}" width="18" height="10" fill="var(--learner)" fill-opacity="0.16"/>'
        + text_at(c.x + 23, ly + 23, "range over three trained seeds")
    )
    body.append(legend(c.x + 290, ly + 20, [("var(--write)", "goal-vector, dial only, vs its oracle dial", "5 3")]))
    return svg(width, ly + 34, "".join(body), "Retargeting cost for every agent")


def figure_metric(rows):
    """Core 2b: the metric itself, one number per agent and goal, on one linear axis.

    The dot is the spec's number, SDL less the null arm's, subtracted per route;
    the bar runs to the per-token subtraction, bracketing the composition bias.
    Where an oracle dial exists, the open diamond is the bias-free version: the
    arm's excess over its oracle less the null arm's excess over its own, so that
    both terms are scored on the routes they encode.
    """
    series = [
        (("bcnv11.s1", "main"), "learner bcnv11", "var(--learner)"),
        (("bcgv11.s1", "main"), "goal-vector, all weights", "var(--write)"),
        (("bcgv11.s1", "dial only"), "goal-vector, dial only", "var(--write)"),
    ]
    series = [s for s in series if runs(rows, *s[0])]
    twins = present(rows, [("bcnv11.s2", "main"), ("bcnv11.s3", "main")])
    taus = sorted({r["tau"] for k, _, _ in series for r in runs(rows, *k)} - {NULL_TAU})
    if not series or not taus:
        return ""

    values = []
    for key, _, _ in series + [(k, None, None) for k in twins]:
        for tau in taus:
            r = run_at(rows, *key, tau)
            if r and r["sdl_minus_null_raw"] is not None:
                values += [r["sdl_minus_null_raw"], r["sdl_minus_null"]]
    lo, hi = min(values + [0]), max(values)
    pad = (hi - lo) * 0.04
    left, top, chart_w, row_h, sub = 112, 26, 640, 46, 12
    c = Chart(left, top, chart_w, len(taus) * row_h, (lo - pad, hi + pad), (0, 1))
    body = []
    ticks = nice_ticks(lo - pad, hi + pad, 6)
    bottom = top + len(taus) * row_h
    for t in ticks:
        x = c.px(t)
        body.append(
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}" stroke="{SOFT if t == 0 else GRID}" '
            f'stroke-width="1"/>'
        )
        body.append(text_at(x, bottom + 16, fmt_bits(t), MONO, 10, SOFT, "middle"))
    body.append(label(left + chart_w / 2, bottom + 34, "retargeting cost at n = 16k (bits)"))
    body.append(text_at(c.px(0) + 4, top - 8, "0 = no cost beyond the reference", SANS, 9.5, SOFT))
    for i, tau in enumerate(taus):
        y0 = top + i * row_h + (row_h - sub * (len(series) - 1)) / 2
        if i:
            body.append(
                f'<line x1="{left - 100}" y1="{top + i * row_h}" x2="{left + chart_w}" y2="{top + i * row_h}" stroke="{GRID}"/>'
            )
        body.append(text_at(left - 100, y0 + sub + 4, f"τ {tau:g}", MONO, 11, INK, weight=600))
        for j, (key, _, colour) in enumerate(series):
            y = y0 + j * sub
            r = run_at(rows, *key, tau)
            if r is None or r["sdl_minus_null_raw"] is None:
                continue
            a, b = r["sdl_minus_null_raw"], r["sdl_minus_null"]
            if j == 0 and twins:
                spread = [x["sdl_minus_null_raw"] for k in [key, *twins] if (x := run_at(rows, *k, tau))]
                body.append(
                    f'<rect x="{c.px(min(spread)):.1f}" y="{y - 6}" width="{c.px(max(spread)) - c.px(min(spread)):.1f}" '
                    f'height="12" fill="{colour}" fill-opacity="0.16"/>'
                )
            body.append(
                f'<line x1="{c.px(a):.1f}" y1="{y}" x2="{c.px(b):.1f}" y2="{y}" stroke="{colour}" stroke-width="3.5" stroke-opacity="0.3"/>'
            )
            if key[1] == "dial only":
                body.append(
                    f'<circle cx="{c.px(a):.1f}" cy="{y}" r="3.6" fill="var(--panel)" stroke="{colour}" stroke-width="1.6"/>'
                )
            else:
                body.append(f'<circle cx="{c.px(a):.1f}" cy="{y}" r="3.6" fill="{colour}"/>')
            o = oracle_based(rows, key, tau)
            if o is not None:
                body.append(marker("diamond", c.px(o), y, colour))
    ly, lx = bottom + 58, left - 100
    items = [
        (lambda x: f'<circle cx="{x}" cy="{ly}" r="3.6" fill="var(--learner)"/>', "learner bcnv11.s1"),
        (
            lambda x: f'<rect x="{x - 9}" y="{ly - 6}" width="18" height="12" fill="var(--learner)" fill-opacity="0.16"/>',
            "range over 3 seeds",
        ),
        (lambda x: f'<circle cx="{x}" cy="{ly}" r="3.6" fill="var(--write)"/>', "goal-vector, all weights"),
        (
            lambda x: f'<circle cx="{x}" cy="{ly}" r="3.6" fill="var(--panel)" stroke="var(--write)" stroke-width="1.6"/>',
            "goal-vector, dial only",
        ),
    ]
    for shape, text in items:
        body.append(shape(lx + 4))
        body.append(text_at(lx + 14, ly + 3.5, text))
        lx += 34 + 5.8 * len(text)
    ly += 22
    lx = left - 100
    body.append(
        f'<line x1="{lx}" y1="{ly}" x2="{lx + 18}" y2="{ly}" stroke="{SOFT}" stroke-width="3.5" stroke-opacity="0.4"/>'
    )
    body.append(text_at(lx + 24, ly + 3.5, "null arm subtracted per route (dot) … per token (bar end)"))
    lx += 360
    body.append(marker("diamond", lx + 4, ly, SOFT))
    body.append(text_at(lx + 14, ly + 3.5, "oracle-based, no composition bias"))
    return svg(790, ly + 14, "".join(body), "The retargeting cost per agent and goal")


VARIANTS = [
    ("main", "reference", "circle-filled"),
    ("ordering 1", "data order", "circle"),
    ("ordering 2", "data order", "circle"),
    ("learner seed 1", "learner seed", "square"),
    ("|V| 256", "|V| 256 not 1024", "triangle"),
    ("lr 1e-5", "lr 1e-5 not 3e-5", "diamond"),
]


def marker(shape, x, y, colour):
    if shape == "circle-filled":
        return f'<circle cx="{x:.1f}" cy="{y}" r="4.5" fill="{colour}"/>'
    if shape == "circle":
        return f'<circle cx="{x:.1f}" cy="{y}" r="3.8" fill="var(--panel)" stroke="{colour}" stroke-width="1.5"/>'
    if shape == "square":
        return f'<rect x="{x - 3.5:.1f}" y="{y - 3.5}" width="7" height="7" fill="var(--panel)" stroke="{colour}" stroke-width="1.5"/>'
    if shape == "triangle":
        return f'<path d="M{x:.1f},{y - 4.5} L{x + 4.2:.1f},{y + 3.5} L{x - 4.2:.1f},{y + 3.5} Z" fill="var(--panel)" stroke="{colour}" stroke-width="1.5"/>'
    if shape == "diamond":
        return f'<path d="M{x:.1f},{y - 4.8} L{x + 4.2:.1f},{y} L{x:.1f},{y + 4.8} L{x - 4.2:.1f},{y} Z" fill="var(--panel)" stroke="{colour}" stroke-width="1.5"/>'
    return f'<path d="M{x - 4:.1f},{y - 4} L{x + 4:.1f},{y + 4} M{x + 4:.1f},{y - 4} L{x - 4:.1f},{y + 4}" stroke="{colour}" stroke-width="1.6"/>'


def figure_noise(rows, tau=2.5):
    """Core 3: the learner's cost at one rate under each replicate setting, and other base seeds."""
    agent = "bcnv11.s1"
    null_main = run_at(rows, agent, "main", NULL_TAU)
    entries = []
    for variant, text, shape in VARIANTS:
        arm, null = run_at(rows, agent, variant, tau), run_at(rows, agent, variant, NULL_TAU)
        if arm and null:
            entries.append((arm["mdl"] - null["mdl"], text, shape, variant))
    for seed in ("bcnv11.s2", "bcnv11.s3"):
        arm, null = run_at(rows, seed, "main", tau), run_at(rows, seed, "main", NULL_TAU)
        if arm and null:
            entries.append((arm["mdl"] - null["mdl"], "another trained base", "cross", seed))
    if len(entries) < 2 or null_main is None:
        return ""
    sweep = [
        (r["sdl_minus_null_raw"], r["tau"])
        for r in runs(rows, agent)
        if r["tau"] != NULL_TAU and r["sdl_minus_null_raw"] is not None
    ]
    values = [e[0] for e in entries] + [v for v, _ in sweep]
    lo, hi = 0, max(values) * 1.12
    width, top, chart_w = 790, 34, 600
    c = Chart(170, top, chart_w, 10, (lo, hi), (0, 1))
    y_row, y_sweep = top + 22, top + 64
    body = []
    for y in (y_row, y_sweep):
        body.append(f'<line x1="{c.x}" y1="{y}" x2="{c.x + chart_w}" y2="{y}" stroke="{GRID}" stroke-width="1"/>')
    body.append(text_at(c.x - 12, y_row + 4, f"one goal, τ {tau:g}, rerun", SANS, 11, INK, "end"))
    body.append(text_at(c.x - 12, y_sweep + 4, "the signal: every τ", SANS, 11, INK, "end"))
    for value, _, shape, _ in sorted(entries, key=lambda e: e[2] == "circle-filled"):  # reference drawn last, on top
        body.append(marker(shape, c.px(value), y_row, "var(--learner)"))
    for i, (value, t) in enumerate(sorted(sweep)):
        body.append(f'<circle cx="{c.px(value):.1f}" cy="{y_sweep}" r="3.6" fill="{tau_colour(t)}"/>')
        body.append(text_at(c.px(value), y_sweep + (-9 if i % 2 else 17), f"{t:g}", MONO, 9, SOFT, "middle"))
    axis_y = y_sweep + 30
    step = 10 ** math.floor(math.log10(hi / 4)) if hi > 0 else 1
    step = step * (2 if hi / step > 10 else 1)
    tick = 0.0
    while tick <= hi:
        body.append(f'<line x1="{c.px(tick):.1f}" y1="{axis_y - 6}" x2="{c.px(tick):.1f}" y2="{axis_y - 2}" stroke="{SOFT}"/>')
        body.append(text_at(c.px(tick), axis_y + 10, fmt_bits(tick), MONO, 10, SOFT, "middle"))
        tick += step
    body.append(label(c.x + chart_w / 2, axis_y + 28, "retargeting cost, SDL less the null arm under the same setting (bits)"))
    seen, lx, ly = set(), c.x, axis_y + 52
    for _, text, shape, _ in entries:
        if text in seen:
            continue
        seen.add(text)
        body.append(marker(shape, lx + 4, ly, SOFT))
        body.append(text_at(lx + 14, ly + 3.5, text))
        lx += 26 + 5.6 * len(text)
    return svg(width, ly + 14, "".join(body), "Noise")


# ----------------------------------------------------------------------
# Appendix figures
# ----------------------------------------------------------------------


def figure_curves(rows, key):
    """Every rate's learning curve for one series."""
    rs = runs(rows, *key)
    if not rs:
        return ""
    width, top, chart_w, chart_h = 790, 30, 520, 200
    n = max(r["n"] for r in rs)
    values = [v for r in rs for v in r["test_bits"]]
    ylo, yhi = math.log10(min(values) * 0.8), math.log10(max(values) * 1.25)
    c = Chart(70, top, chart_w, chart_h, (math.log10(0.5), math.log10(n * 1.3)), (ylo, yhi), xlog=True, ylog=True)
    body = [
        c.frame(
            [(0.5, "0"), (1, "1"), (16, "16"), (256, "256"), (4096, "4k"), (n, fmt_bits(n))],
            log_ticks(10**ylo, 10**yhi, [0.1, 0.2, 0.3, 0.5, 1, 2, 5, 10, 20, 50, 100]),
            xlabel="demonstrations seen",
            ylabel="bits per route on T",
        )
    ]
    for r in rs:
        pts = [(max(t, 0.5), v) for t, v in zip(r["bounds"], r["test_bits"])]
        body.append(c.line(pts, tau_colour(r["tau"]), width=2.2 if r["tau"] == NULL_TAU else 1.6, r=1.7))
    items = [(tau_colour(r["tau"]), f"τ {r['tau']:g}" + (" null" if r["tau"] == NULL_TAU else ""), None) for r in rs]
    body.append(legend(70, top + chart_h + 58, items, step=82))
    return svg(width, top + chart_h + 70, "".join(body), f"Learning curves, {SERIES[key][0]}")


def figure_sweep(rows):
    """Total SDL across the sweep for every series, log scale."""
    keys = present(rows, list(SERIES))
    if not keys:
        return ""
    width, top, chart_w, chart_h = 790, 30, 560, 220
    values = [r["mdl"] for k in keys for r in runs(rows, *k)]
    taus = sorted({r["tau"] for r in rows})
    lo, hi = math.log10(min(values) * 0.6), math.log10(max(values) * 1.8)
    c = Chart(80, top, chart_w, chart_h, (min(taus) - 1, max(taus) + 1), (lo, hi), ylog=True)
    body = [
        c.frame(
            [(t, f"{t:g}") for t in taus],
            log_ticks(10**lo, 10**hi, [10, 100, 1000, 3000, 10_000, 30_000, 100_000, 300_000, 1_000_000]),
            xlabel="exchange rate of the goal, τ (steps)",
            ylabel="SDL (bits)",
        )
    ]
    for key in keys:
        name, colour, dash = SERIES[key]
        body.append(
            c.line(
                [(r["tau"], r["mdl"]) for r in runs(rows, *key)],
                colour,
                width=1.2 if key[0] in ("bcnv11.s2", "bcnv11.s3") else 1.8,
                dash=dash,
                r=2,
            )
        )
    items = [(SERIES[k][1], SERIES[k][0], SERIES[k][2]) for k in keys if k[0] not in ("bcnv11.s2", "bcnv11.s3")]
    body.append(legend(80, top + chart_h + 62, items[:3]))
    if len(items) > 3:
        body.append(legend(80, top + chart_h + 82, items[3:]))
    return svg(width, top + chart_h + 96, "".join(body), "SDL across the sweep")


def figure_split(rows):
    """SDL less the null, split into levels whose choice the goal flips and the rest, per competent series."""
    keys = [k for k in present(rows, CORE_SERIES) if competent(rows, *k)]
    if not keys:
        return ""
    width, top, row_h, bar_w = 790, 24, 15, 190
    peak = max(r["mdl"] for k in keys for r in runs(rows, *k))
    body = []
    for column, key in enumerate(keys):
        cx = 60 + column * 255
        body.append(text_at(cx - 40, top, SERIES[key][0], SANS, 11, INK, weight=600))
        for i, r in enumerate(runs(rows, *key)):
            yy = top + 12 + i * (row_h + 5)
            flipped = r["cumulative_flipped"][-1]
            wf, wr = bar_w * flipped / peak, bar_w * (r["mdl"] - flipped) / peak
            body.append(text_at(cx - 6, yy + 11, f"τ {r['tau']:g}", MONO, 9.5, SOFT, "end"))
            body.append(
                f'<rect x="{cx}" y="{yy}" width="{wf:.1f}" height="{row_h}" fill="{tau_colour(r["tau"]) if r["tau"] != NULL_TAU else SOFT}"/>'
            )
            body.append(f'<rect x="{cx + wf:.1f}" y="{yy}" width="{wr:.1f}" height="{row_h}" fill="var(--grid-strong)"/>')
            body.append(text_at(cx + wf + wr + 4, yy + 11, fmt_bits(round(r["mdl"])), MONO, 9.5, INK))
    rows_n = max(len(runs(rows, *k)) for k in keys)
    bottom = top + 12 + rows_n * (row_h + 5) + 16
    body.append(
        f'<rect x="60" y="{bottom - 6}" width="14" height="10" fill="var(--poor)"/>'
        + text_at(80, bottom + 3, "levels the goal flips")
    )
    body.append(
        f'<rect x="230" y="{bottom - 6}" width="14" height="10" fill="var(--grid-strong)"/>'
        + text_at(250, bottom + 3, "levels it leaves alone")
    )
    return svg(width, bottom + 12, "".join(body), "Where the bits go")


def figure_steps(rows):
    """The step early stopping chose, against t, at tau 2.5 and the null, for the core series."""
    keys = present(rows, CORE_SERIES)
    chosen = [(k, run_at(rows, *k, tau)) for k in keys for tau in (2.5, NULL_TAU)]
    chosen = [(k, r) for k, r in chosen if r]
    if not chosen:
        return ""
    n = max(r["n"] for _, r in chosen)
    steps = [s for _, r in chosen for s in r["steps_run"]]
    width, top, chart_w, chart_h = 790, 30, 520, 190
    c = Chart(
        80,
        top,
        chart_w,
        chart_h,
        (0, math.log10(n * 1.3)),
        (math.log10(0.7), math.log10(max(steps + [10]) * 1.5)),
        xlog=True,
        ylog=True,
    )
    body = [
        c.frame(
            [(1, "1"), (16, "16"), (256, "256"), (4096, "4k"), (n, fmt_bits(n))],
            log_ticks(1, max(steps + [10]) * 1.5, [1, 3, 10, 30, 100, 300, 1000, 3000]),
            xlabel="demonstrations seen",
            ylabel="chosen step",
        )
    ]
    for key, r in chosen:
        name, colour, dash = SERIES[key]
        pts = [(t, max(s, 0.8)) for t, s in zip(r["bounds"], r["best_step"]) if t > 0]
        body.append(c.line(pts, colour if r["tau"] != NULL_TAU else SOFT, width=1.5, dash=dash, r=1.8))
    body.append(text_at(c.x + 4, c.py(0.8) - 5, "0.8 = the base kept", SANS, 9))
    body.append(
        legend(
            80,
            top + chart_h + 58,
            [(SERIES[k][1], SERIES[k][0], SERIES[k][2]) for k in keys] + [(SOFT, "null arms", None)],
            step=200,
        )
    )
    return svg(width, top + chart_h + 72, "".join(body), "Early stopping")


def figure_convergence(rows):
    """Cost so far as n grows, against each agent's proper reference, signed and linear.

    The learner against its null arm (per route, as in the headline); the
    goal-vector agent against its oracle dial on the same stream, which needs
    no null arm.
    """
    keys = [k for k in present(rows, CORE_SERIES) if competent(rows, *k)]
    if not keys:
        return ""
    width, top, chart_w, chart_h = 790, 30, 200, 180
    body = []
    for i, key in enumerate(keys):
        rs = [r for r in runs(rows, *key) if r["tau"] != NULL_TAU]
        null = run_at(rows, *key, NULL_TAU)
        curves = []
        for r in rs:
            if "oracle_cumulative" in r:
                reference, what = r["oracle_cumulative"], "oracle dial"
            elif null:
                reference, what = null["cumulative"], "null arm"
            else:
                continue
            curves.append((r, [(t, a - b) for t, a, b in zip(r["bounds"], r["cumulative"], reference) if t > 0]))
        if not curves:
            continue
        n = rs[0]["n"]
        values = [v for _, pts in curves for _, v in pts] + [0]
        lo, hi = min(values) * 1.1, max(values) * 1.1
        x = 80 + i * 245
        c = Chart(x, top + 16, chart_w, chart_h, (0, math.log10(n * 1.3)), (lo, hi), xlog=True)
        span = hi - lo
        unit = 1000 if span > 3000 else 500 if span > 1500 else 200
        ticks = [(v, fmt_bits(v)) for v in range(int(math.ceil(lo / unit) * unit), int(hi) + 1, unit)]
        body.append(text_at(x, top + 4, f"{PANEL_TITLE[key]} − {what}", SANS, 10.5, INK, weight=600))
        body.append(
            c.frame(
                [(1, "1"), (64, "64"), (4096, "4k"), (n, fmt_bits(n))],
                ticks,
                xlabel="n",
                ylabel="cost so far (bits)" if i == 0 else None,
            )
        )
        body.append(
            f'<line x1="{c.x}" y1="{c.py(0):.1f}" x2="{c.x + chart_w}" y2="{c.py(0):.1f}" stroke="{SOFT}" stroke-width="1"/>'
        )
        for r, pts in curves:
            body.append(c.line(pts, tau_colour(r["tau"]), width=1.4, dots=False))
    return svg(width, top + chart_h + 66, "".join(body), "Cost as n grows")


def table_html(rows):
    head = "<tr><th>agent</th><th>learner</th><th>τ</th><th>flipped</th><th>L(0)</th><th>L(n)</th><th>SDL</th><th>SDL − null</th><th>ceiling − null</th><th>EDL</th></tr>"
    body = []
    order = list(SERIES)
    for r in sorted(
        (r for r in rows if (r["agent"], r["variant"]) in SERIES),
        key=lambda r: (order.index((r["agent"], r["variant"])), r["tau"]),
    ):
        diff = "" if r["sdl_minus_null"] is None or r["tau"] == NULL_TAU else f"{r['sdl_minus_null']:,.0f}"
        ceil = "" if r["ceiling_minus_null"] is None or r["tau"] == NULL_TAU else f"{r['ceiling_minus_null']:,.0f}"
        body.append(
            f"<tr><td>{r['agent']}</td><td>{'dial only' if r['variant'] == 'dial only' else 'all weights'}</td><td>{r['tau']:g}</td>"
            f"<td>{r['flipped_fraction']:.0%}</td><td>{r['test_bits'][0]:.2f}</td><td>{r['test_bits'][-1]:.2f}</td>"
            f"<td>{r['mdl']:,.0f}</td><td>{diff}</td><td>{ceil}</td><td>{r['edl']:,.0f}</td></tr>"
        )
    return f'<div class="scroll"><table>{head}{"".join(body)}</table></div>'


# ----------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------


def section(title, figure, note, figure_id):
    if not figure and not note:
        return ""
    return f'<section id="{figure_id}"><h3>{title}</h3>{note}<div class="scroll">{figure}</div></section>'


def page(data, notes):
    rows = data["rows"]
    floor_bits = math.log2(max(len({r["tau"] for r in rows}), 2))
    core = [
        ("1 · The retargeting cost, every agent", figure_overview(rows), "overview"),
        ("2 · How fast each agent reaches its reference", figure_all_agents(rows), "all-agents"),
        ("3 · Per goal, and how much the reference matters", figure_metric(rows), "metric"),
        ("4 · Is the signal bigger than the noise?", figure_noise(rows), "noise"),
    ]
    appendix = [
        ("A00 · The learner's cost against the size of the change, log scale", figure_headline(rows, floor_bits), "headline"),
        ("A0 · One goal, three learners, raw loss", figure_close_up(rows, 2.5), "close-up"),
        ("A1 · Learning curves, learner", figure_curves(rows, ("bcnv11.s1", "main")), "curves-learner"),
        ("A2 · Learning curves, goal-vector agent", figure_curves(rows, ("bcgv11.s1", "main")), "curves-goalvec"),
        (
            "A3 · Learning curves, goal-vector agent, dial-only learner",
            figure_curves(rows, ("bcgv11.s1", "dial only")),
            "curves-dial",
        ),
        ("A4 · Learning curves, memoriser", figure_curves(rows, ("bcov11.n1k.s1", "main")), "curves-memoriser"),
        ("A5 · Total cost across the sweep", figure_sweep(rows), "sweep"),
        ("A6 · Where the bits go", figure_split(rows), "split"),
        ("A7 · Cost as n grows", figure_convergence(rows), "convergence"),
        ("A8 · What early stopping chose", figure_steps(rows), "steps"),
    ]
    note = lambda key: "".join(f"<p>{p}</p>" for p in notes.get(key, []))
    return TEMPLATE.format(
        intro=note("intro"),
        takeaways="".join(f"<li>{t}</li>" for t in notes.get("takeaways", [])),
        core="".join(section(t, f, note(k), k) for t, f, k in core),
        appendix_intro=note("appendix"),
        appendix="".join(section(t, f, note(k), k) for t, f, k in appendix),
        table=table_html(rows),
        method=note("method"),
    )


TEMPLATE = """<title>Retargeting Cost Pilot</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;600&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&display=swap">
<style>
/* One reading column: takeaways, three core figures, then a quieter appendix. */
:root {{
  --panel: #fdfcf9; --ink: #23252b; --ink-soft: #5c5f6a;
  --grid: #ebe8e1; --grid-strong: #d6d2c8; --wash: #f4f2ec;
  --rich: #a63d2f; --poor: #2f5a78; --write: #6f5f9c;
  --learner: #2f5a78; --memoriser: #a63d2f;
  --sans: 'IBM Plex Sans', system-ui, sans-serif;
  --serif: 'Source Serif 4', Georgia, serif;
  --mono: 'IBM Plex Mono', ui-monospace, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --panel: #20232b; --ink: #e8e6df; --ink-soft: #9a9da8; --grid: #2c3039; --grid-strong: #434855; --wash: #262a33;
  --rich: #f08c3a; --poor: #4ea6dd; --write: #a596d6; --learner: #4ea6dd; --memoriser: #f08c3a;
  color-scheme: dark; }} }}
:root[data-theme="dark"] {{
  --panel: #20232b; --ink: #e8e6df; --ink-soft: #9a9da8; --grid: #2c3039; --grid-strong: #434855; --wash: #262a33;
  --rich: #f08c3a; --poor: #4ea6dd; --write: #a596d6; --learner: #4ea6dd; --memoriser: #f08c3a;
  color-scheme: dark; }}
body {{ background: var(--panel); color: var(--ink); font-family: var(--serif); font-size: 17px; line-height: 1.55; }}
main {{ max-width: 820px; margin: 0 auto; padding-inline: 16px; padding-block: 40px 72px; }}
h1 {{ font-family: var(--sans); font-weight: 600; font-size: 27px; line-height: 1.2; margin: 0 0 14px; text-wrap: balance; }}
h2 {{ font-family: var(--sans); font-weight: 600; font-size: 12px; letter-spacing: 0.12em; text-transform: uppercase; color: var(--ink-soft); margin: 52px 0 4px; padding-top: 14px; border-top: 1px solid var(--grid-strong); }}
h3 {{ font-family: var(--sans); font-weight: 600; font-size: 16px; margin: 34px 0 6px; text-wrap: balance; }}
p {{ max-width: 66ch; margin: 0 0 12px; }}
.kicker {{ font-family: var(--sans); font-size: 11px; letter-spacing: 0.12em; text-transform: uppercase; color: var(--ink-soft); margin: 0 0 10px; }}
.takeaways {{ background: var(--wash); padding: 16px 20px 6px 36px; margin: 20px 0 8px; max-width: 70ch; }}
.takeaways li {{ margin-bottom: 10px; }}
.appendix h3 {{ font-size: 14px; color: var(--ink-soft); }}
.appendix p {{ font-size: 15px; }}
.scroll {{ overflow-x: auto; margin-top: 8px; }}
table {{ border-collapse: collapse; font-family: var(--mono); font-size: 12px; font-variant-numeric: tabular-nums; min-width: 700px; }}
th {{ font-family: var(--sans); font-weight: 600; font-size: 10.5px; letter-spacing: 0.06em; text-transform: uppercase; color: var(--ink-soft); text-align: right; padding: 6px 8px; border-bottom: 1px solid var(--grid-strong); }}
td {{ text-align: right; padding: 4px 8px; border-bottom: 1px solid var(--grid); }}
th:nth-child(-n+2), td:nth-child(-n+2) {{ text-align: left; }}
code {{ font-family: var(--mono); font-size: 0.85em; }}
</style>
<main>
<p class="kicker">4YP · retargeting cost · route-model pilot · 2026-10-10</p>
<h1>Does the retargeting cost find the dial?</h1>
{intro}
<ul class="takeaways">{takeaways}</ul>
<h2>Core results</h2>
{core}
<div class="appendix">
<h2>Appendix: controls and exploration</h2>
{appendix_intro}
{appendix}
<section id="table"><h3>A9 · Every reference run</h3><div class="scroll">{table}</div></section>
<section id="method"><h3>A10 · Method, exactly</h3>{method}</section>
</div>
</main>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=pathlib.Path, default=SRC / "data.json")
    parser.add_argument("--notes", type=pathlib.Path, default=SRC / "notes.json")
    parser.add_argument("--out", type=pathlib.Path, default=SRC / "report.html")
    args = parser.parse_args()
    data = json.loads(args.data.read_text())
    notes = json.loads(args.notes.read_text()) if args.notes.exists() else {}
    args.out.write_text(page(data, notes))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
