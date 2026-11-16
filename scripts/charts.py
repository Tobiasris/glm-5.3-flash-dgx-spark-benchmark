#!/usr/bin/env python3
"""Build the report's SVG charts from results/*.json. Standard library only. Run from the report folder:
  python3 scripts/charts.py
"""
import csv, datetime, json, statistics as st

R = "results/"
C = {"vllm": "#8a8f98", "tf1104": "#3b82c4", "tfpy": "#e07b39", "tf1104-q4": "#7fb0dd", "tf1104-mtp": "#b9c6d6"}
N = {"vllm": "vLLM (NVFP4 + MTP)", "tf1104": "TensorFold 1104", "tfpy": "TensorFold Python",
     "tf1104-q4": "1104, dense q4", "tf1104-mtp": "1104, MTP drafter"}
r2 = {k: json.load(open(f"{R}r2-{k}.json")) for k in C}
FONT = 'font-family="Helvetica,Arial,sans-serif"'


def svg(w, h, title, sub, body):
    return "\n".join([f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" {FONT}>',
                      f'<rect width="{w}" height="{h}" fill="#fff"/>',
                      f'<text x="24" y="30" font-size="17" font-weight="bold" fill="#111">{title}</text>',
                      f'<text x="24" y="49" font-size="12" fill="#555">{sub}</text>'] + body + ["</svg>"])


def axis(L, T, B, R_, mx, step, label):
    o = []
    v = 0
    while v <= mx + 1e-9:
        y = B - (B - T) * v / mx
        o.append(f'<line x1="{L}" x2="{R_}" y1="{y:.1f}" y2="{y:.1f}" stroke="#e6e8eb"/>'
                 f'<text x="{L-8}" y="{y+4:.1f}" font-size="11" text-anchor="end" fill="#555">{v:g}</text>')
        v += step
    o.append(f'<text x="{L-44}" y="{(T+B)/2}" font-size="11" fill="#555" transform="rotate(-90 {L-44} {(T+B)/2})" text-anchor="middle">{label}</text>')
    return o


def legend(keys, x, y, dx=190):
    o = []
    for i, k in enumerate(keys):
        o.append(f'<rect x="{x+i*dx}" y="{y}" width="12" height="12" rx="2" fill="{C[k]}"/>'
                 f'<text x="{x+i*dx+18}" y="{y+11}" font-size="12" fill="#111">{N[k]}</text>')
    return o


def bars(groups, keys, values, mx, step, title, sub, ylab, fn, errs=None, w=780, h=430, fmt="{:.0f}"):
    L, T, B, Rr = 80, 75, 330, w - 20
    o = axis(L, T, B, Rr, mx, step, ylab)
    gw = (Rr - L) / len(groups); bw = min(46, (gw - 30) / len(keys))
    for g, lab in enumerate(groups):
        x0 = L + g * gw + (gw - bw * len(keys) - 4 * (len(keys) - 1)) / 2
        for i, k in enumerate(keys):
            v = values[k][g]
            if v is None:
                continue
            x = x0 + i * (bw + 4); hh = (B - T) * v / mx
            o.append(f'<rect x="{x:.1f}" y="{B-hh:.1f}" width="{bw:.1f}" height="{hh:.1f}" rx="2" fill="{C[k]}"/>')
            top = B - hh
            if errs and errs[k][g]:
                lo, hi = errs[k][g]
                y1, y2 = B - (B - T) * lo / mx, B - (B - T) * hi / mx; cx = x + bw / 2
                o.append(f'<line x1="{cx}" x2="{cx}" y1="{y1:.1f}" y2="{y2:.1f}" stroke="#222" stroke-width="1.2"/>'
                         f'<line x1="{cx-5}" x2="{cx+5}" y1="{y2:.1f}" y2="{y2:.1f}" stroke="#222"/>'
                         f'<line x1="{cx-5}" x2="{cx+5}" y1="{y1:.1f}" y2="{y1:.1f}" stroke="#222"/>')
                top = min(top, y2)
            o.append(f'<text x="{x+bw/2:.1f}" y="{top-5:.1f}" font-size="11" text-anchor="middle" fill="#111">{fmt.format(v)}</text>')
        o.append(f'<text x="{L+g*gw+gw/2:.1f}" y="{B+20}" font-size="13" text-anchor="middle" fill="#111">{lab}</text>')
    o += legend(keys, L, B + 42, dx=(Rr - L) / len(keys))
    open(fn, "w").write(svg(w, h, title, sub, o)); print("wrote", fn)


def mixed(k, part="mixed"):
    return r2[k].get(part, {})


# 1. Throughput curve, mixed prompts, with min-max whiskers
def curve():
    w, h = 780, 440; L, T, B, Rr = 80, 75, 340, 600
    lv = [1, 2, 4, 8]; xs = {l: L + (Rr - L) * i / 3 for i, l in enumerate(lv)}
    mx = 110
    o = axis(L, T, B, Rr, mx, 20, "total tok/s")
    for l in lv:
        o.append(f'<text x="{xs[l]}" y="{B+20}" font-size="12" text-anchor="middle" fill="#111">{l}</text>')
    o.append(f'<text x="{(L+Rr)/2}" y="{B+40}" font-size="12" text-anchor="middle" fill="#555">concurrent requests</text>')
    for k in ["vllm", "tf1104", "tfpy"]:
        m = mixed(k); pts = [(xs[int(l)], v["total_tok_s"]) for l, v in m.items()]
        o.append('<polyline fill="none" stroke="%s" stroke-width="2.5" points="%s"/>' % (
            C[k], " ".join(f"{x:.1f},{B-(B-T)*s['mean']/mx:.1f}" for x, s in pts)))
        for x, s in pts:
            y = B - (B - T) * s["mean"] / mx
            y1, y2 = B - (B - T) * s["min"] / mx, B - (B - T) * s["max"] / mx
            o.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{y1:.1f}" y2="{y2:.1f}" stroke="{C[k]}" stroke-width="1.5"/>'
                     f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{C[k]}"/>')
        x, s = pts[-1]
        o.append(f'<text x="{x+10:.1f}" y="{B-(B-T)*s["mean"]/mx+4:.1f}" font-size="12" fill="{C[k]}" font-weight="bold">{s["mean"]:.0f} {N[k]}</text>')
    open("fig1_throughput_curve.svg", "w").write(svg(w, h, "Total decode throughput vs. concurrency",
         "mixed prose/Q&amp;A prompts, T=0.7, 400 tokens per request, 5 runs per point (whiskers: min to max)", o))
    print("wrote fig1")


# 2. Prose vs. code
def code():
    keys = ["vllm", "tf1104-mtp", "tf1104", "tfpy"]
    vals = {k: [mixed(k)["1"]["total_tok_s"]["mean"], mixed(k, "code")["1"]["total_tok_s"]["mean"],
                mixed(k)["4"]["total_tok_s"]["mean"], mixed(k, "code")["4"]["total_tok_s"]["mean"]] for k in keys}
    errs = {k: [(s["min"], s["max"]) for s in [mixed(k)["1"]["total_tok_s"], mixed(k, "code")["1"]["total_tok_s"],
                                                mixed(k)["4"]["total_tok_s"], mixed(k, "code")["4"]["total_tok_s"]]] for k in keys}
    bars(["prose, 1 req", "code, 1 req", "prose, 4 req", "code, 4 req"], keys, vals, 120, 20,
         "Prose vs. code: speculation pays more on code", "total tok/s, mean of 5 (prose) or 3 (code) runs, whiskers min to max",
         "total tok/s", "fig2_prose_vs_code.svg", errs)


# 3. Where the gain comes from (4 concurrent, mixed)
def decomposition():
    steps = [("vllm", "vLLM + MTP"), ("tf1104-mtp", "TensorFold + MTP"), ("tf1104", "+ DFlash2 drafter"),
             ("tf1104-q4", "+ dense weights q4"), ("tfpy", "Python recipe")]
    w, h = 780, 420; L, T, B, Rr = 80, 75, 320, 760; mx = 100
    o = axis(L, T, B, Rr, mx, 20, "total tok/s at 4 concurrent")
    bw = 96; gap = (Rr - L - bw * len(steps)) / len(steps); prev = None
    for i, (k, lab) in enumerate(steps):
        v = mixed(k)["4"]["total_tok_s"]["mean"]; x = L + gap / 2 + i * (bw + gap); hh = (B - T) * v / mx
        o.append(f'<rect x="{x:.1f}" y="{B-hh:.1f}" width="{bw}" height="{hh:.1f}" rx="2" fill="{C[k]}"/>'
                 f'<text x="{x+bw/2:.1f}" y="{B-hh-22:.1f}" font-size="13" font-weight="bold" text-anchor="middle" fill="#111">{v:.1f}</text>')
        if prev is not None:
            d = (v / prev - 1) * 100
            o.append(f'<text x="{x+bw/2:.1f}" y="{B-hh-7:.1f}" font-size="11" text-anchor="middle" fill="#2a7a2a">{d:+.0f} %</text>')
        prev = v
        o.append(f'<text x="{x+bw/2:.1f}" y="{B+18}" font-size="11" text-anchor="middle" fill="#111">{lab}</text>')
    o.append(f'<text x="{L}" y="{B+52}" font-size="11" fill="#555">Steps 2 to 4 change one setting at a time on the 1104 recipe. The Python recipe is a different engine version (v0.6.0) at its defaults.</text>')
    open("fig3_where_the_gain_comes_from.svg", "w").write(svg(w, h, "Where the gain comes from",
         "mixed prompts, 4 concurrent requests, mean of 5 runs; percentages vs. the bar to the left", o))
    print("wrote fig3")


# 4. Benchmark trap: identical prompts vs. mixed prompts (round 1 data)
def trap():
    f = {k: json.load(open(f"{R}fenster-{n}.json")) for k, n in [("vllm", "tf-ausgang-vllm"), ("tf1104", "tf1104"), ("tfpy", "tfpy")]}
    g = {k: json.load(open(f"{R}gemischt-{k}.json")) for k in ["vllm", "tf1104", "tfpy"]}
    def at(d, c): return next(s["gesamt_tok_s"] for s in d["stufen"] if s["parallel"] == c)
    keys = ["vllm", "tf1104", "tfpy"]
    vals = {k: [at(f[k], 4), g[k]["stufen"]["4"]["gesamt_tok_s"]] for k in keys}
    bars(["4 identical prompts, T=0", "4 different prompts, T=0.7"], keys, vals, 160, 20,
         "Benchmark trap: identical prompts inflate TensorFold by 70-80 %",
         "total tok/s at 4 concurrent requests; copy drafts let identical greedy streams draft from each other",
         "total tok/s", "fig4_benchmark_trap.svg")


# 5. Decode next to a long prefill
def interleave():
    keys = ["vllm", "tf1104", "tfpy"]
    m = {k: r2[k]["interleave"] for k in keys}
    avg = lambda k, f: st.mean(x[f] for x in m[k] if x[f] is not None)
    vals = {k: [avg(k, "decode_before_tok_s"), avg(k, "decode_during_tok_s"), avg(k, "decode_after_tok_s")] for k in keys}
    bars(["before the long prompt", "while 50k tokens prefill", "after"], keys, vals, 60, 10,
         "A chat stream next to a 50k-token prefill",
         "decode tok/s of one streaming reply while a second request sends a cold 50k-token prompt; mean of 2 runs",
         "tok/s of the chat stream", "fig5_decode_beside_prefill.svg", fmt="{:.1f}")


# 6. Energy: tokens per joule
def energy():
    def pw(fn):
        out = []
        for r in csv.reader(open(fn)):
            try:
                out.append((datetime.datetime.strptime(r[0].strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp(), float(r[1])))
            except (ValueError, IndexError):
                pass
        return out
    H, W = pw(R + "r2-power-head.csv"), pw(R + "r2-power-worker.csv")
    def p(s, a, b): return st.mean(x for t, x in s if a <= t <= b)
    keys = ["vllm", "tf1104", "tfpy"]; vals = {}; watts = {}
    for k in keys:
        vals[k] = []; watts[k] = []
        for l in ["1", "4"]:
            v = mixed(k)[l]; ww = p(H, v["t_start"], v["t_end"]) + p(W, v["t_start"], v["t_end"])
            vals[k].append(v["total_tok_s"]["mean"] / ww); watts[k].append(ww)
    bars(["1 request", "4 concurrent"], keys, vals, 0.8, 0.2, "Energy: tokens per joule (both Sparks)",
         "total tok/s divided by the summed GPU power of both Sparks (nvidia-smi power.draw, 2 s samples)",
         "tokens per joule", "fig6_tokens_per_joule.svg", fmt="{:.2f}")


curve(); code(); decomposition(); trap(); interleave(); energy()
