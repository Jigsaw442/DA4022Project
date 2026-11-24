"""Collect all metrics into tables (Markdown + LaTeX) and draw the report figures.

    python scripts/make_report_assets.py --results results
Writes results/tables/*.md|*.tex, results/figures/*.png|*.pdf and results/summary.json.
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DECODE_LABEL = {"greedy": "Greedy", "adhh": "AD-HH", "vrhh": "VR-HH (ours)", "tfhh": "TF-HH (ours, 64-tok)",
                "vrhh-transfer": "VR-HH, heads from 576"}
COLORS = {"greedy": "#4C72B0", "adhh": "#DD8452", "vrhh": "#55A868", "tfhh": "#C44E52"}


def load(path):
    try:
        return json.load(open(path))
    except FileNotFoundError:
        return None


def collect(res):
    rows = {}
    for kind in ("caption", "pope"):
        for f in glob.glob(os.path.join(res, kind, "*", "metrics.json")):
            name = os.path.basename(os.path.dirname(f))
            rows.setdefault(name, {})[kind] = load(f)
    return rows


def fmt(x, nd=1):
    return "–" if x is None else f"{x:.{nd}f}"


def table(rows, names, title, path):
    hdr = ["Config", "Vis. tok", "CHAIR_s ↓", "CHAIR_i ↓", "Recall ↑", "Len", "POPE Acc ↑", "POPE F1 ↑",
           "Yes %", "s/caption"]
    lines_md = [f"### {title}", "", "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    lines_tex = [r"\begin{tabular}{l" + "r" * (len(hdr) - 1) + "}", r"\toprule",
                 " & ".join(h.replace("↓", r"$\downarrow$").replace("↑", r"$\uparrow$").replace("_", r"\_")
                            .replace("%", r"\%") for h in hdr) + r" \\", r"\midrule"]
    for n in names:
        if n not in rows:
            continue
        c, p = rows[n].get("caption") or {}, rows[n].get("pope") or {}
        tok = c.get("avg_visual_tokens", p.get("avg_visual_tokens"))
        vals = [n, fmt(tok, 0), fmt(c.get("CHAIR_s")), fmt(c.get("CHAIR_i")), fmt(c.get("recall")),
                fmt(c.get("avg_len_words")), fmt(p.get("accuracy")), fmt(p.get("f1")), fmt(p.get("yes_ratio")),
                fmt(c.get("sec_per_caption"), 2)]
        lines_md.append("| " + " | ".join(vals) + " |")
        lines_tex.append(" & ".join(v.replace("_", r"\_") for v in vals) + r" \\")
    lines_tex += [r"\bottomrule", r"\end{tabular}"]
    open(path + ".md", "w").write("\n".join(lines_md) + "\n")
    open(path + ".tex", "w").write("\n".join(lines_tex) + "\n")
    return "\n".join(lines_md)


def savefig(fig, figdir, name):
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, name + ".png"), dpi=200)
    fig.savefig(os.path.join(figdir, name + ".pdf"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    res = args.results
    tdir, fdir = os.path.join(res, "tables"), os.path.join(res, "figures")
    os.makedirs(tdir, exist_ok=True); os.makedirs(fdir, exist_ok=True)
    rows = collect(res)
    summary = {}
    budgets = [("none576", 576), ("hybrid144", 144), ("hybrid64", 64), ("hybrid16", 16)]

    # ---------- tables
    main_names = [f"{n}_{d}" for n, _ in budgets for d in ("greedy", "adhh", "vrhh")]
    main_names += ["hybrid64_vrhh-transfer", "adaptive16-144_greedy", "adaptive16-144_vrhh",
                   "hybrid64_tfhh", "hybrid16_tfhh"]
    md = [table(rows, main_names, "Main results (LLaVA-1.5-7B)", os.path.join(tdir, "main_results"))]
    md.append(table(rows, ["none576_greedy", "pool64_greedy", "topk64_greedy", "hybrid64_greedy"],
                    "Compressor comparison at 64 visual tokens (greedy)", os.path.join(tdir, "compressors")))
    print("\n\n".join(md))
    summary["metrics"] = rows

    # ---------- Fig 1: CHAIR_s / POPE-F1 vs visual tokens
    for metric, kind, fname, ylabel in (("CHAIR_s", "caption", "chair_vs_tokens", "CHAIR$_s$ (lower is better)"),
                                        ("f1", "pope", "pope_vs_tokens", "POPE-adversarial F1")):
        fig, ax = plt.subplots(figsize=(4.6, 3.3))
        any_pts = False
        for d in ("greedy", "adhh", "vrhh"):
            xs, ys = [], []
            for n, b in budgets:
                m = (rows.get(f"{n}_{d}") or {}).get(kind)
                if m:
                    xs.append(b); ys.append(m[metric])
            if xs:
                any_pts = True
                ax.plot(xs, ys, "o-", color=COLORS[d], label=DECODE_LABEL[d])
        for n, mk, lab in (("adaptive16-144_vrhh", "*", "Adaptive 16/144 + VR-HH"),
                           ("adaptive16-144_greedy", "x", "Adaptive 16/144"),
                           ("hybrid64_tfhh", "D", DECODE_LABEL["tfhh"])):
            m = (rows.get(n) or {}).get(kind)
            if m:
                ax.scatter([m["avg_visual_tokens"]], [m[metric]], marker=mk, s=80, color="k" if mk != "D" else COLORS["tfhh"],
                           label=lab, zorder=5)
        if any_pts:
            ax.set_xscale("log", base=2); ax.set_xticks([16, 64, 144, 576]); ax.set_xticklabels(["16", "64", "144", "576"])
            ax.set_xlabel("visual tokens"); ax.set_ylabel(ylabel); ax.grid(alpha=.3); ax.legend(fontsize=7)
            savefig(fig, fdir, fname)
        else:
            plt.close(fig)

    # ---------- Fig 2: diagnosis — attention mass vs compression
    heads = {n: load(os.path.join(res, "heads", f"{n}.json")) for n, _ in budgets}
    heads = {k: v for k, v in heads.items() if v}
    if heads:
        fig, axs = plt.subplots(1, 2, figsize=(9, 3.3))
        diag = {}
        for n, b in budgets:
            h = heads.get(n)
            if not h:
                continue
            t, v = np.array(h["text_mass"]), np.array(h["vis_mass"])
            col = {576: "#4C72B0", 144: "#55A868", 64: "#DD8452", 16: "#C44E52"}.get(b)
            axs[0].plot(t.mean(1), label=f"text, {b} tok", ls="-", color=col)
            axs[0].plot(v.mean(1), label=f"visual, {b} tok", ls="--", color=col)
            top = h["top"][:20]
            tt = np.mean([t[l][hh] for l, hh in top]); vv = np.mean([v[l][hh] for l, hh in top])
            rest = np.ones_like(t, dtype=bool)
            for l, hh in top:
                rest[l][hh] = False
            diag[b] = {"hall_heads_text": float(tt), "hall_heads_vis": float(vv), "hall_ratio": float(tt / max(vv, 1e-6)),
                       "other_heads_text": float(t[rest].mean()), "other_heads_vis": float(v[rest].mean()),
                       "other_ratio": float(t[rest].mean() / max(v[rest].mean(), 1e-6))}
        axs[0].set_xlabel("layer"); axs[0].set_ylabel("mean attention mass"); axs[0].legend(fontsize=6); axs[0].grid(alpha=.3)
        bs = sorted(diag, reverse=True)
        x = np.arange(len(bs))
        axs[1].bar(x - .2, [diag[b]["hall_ratio"] for b in bs], .4, label="top-20 hallucination heads", color="#C44E52")
        axs[1].bar(x + .2, [diag[b]["other_ratio"] for b in bs], .4, label="other heads", color="#8C8C8C")
        axs[1].set_xticks(x); axs[1].set_xticklabels([str(b) for b in bs]); axs[1].set_xlabel("visual tokens")
        axs[1].set_ylabel("text / visual attention ratio"); axs[1].legend(fontsize=7); axs[1].grid(alpha=.3, axis="y")
        savefig(fig, fdir, "diagnosis_attention")
        summary["diagnosis"] = diag

        # Fig 3: contrastive-score heatmaps + overlap of top-20 sets
        fig, axs = plt.subplots(1, len(heads), figsize=(3.2 * len(heads), 3))
        axs = np.atleast_1d(axs)
        for ax, (n, b) in zip(axs, [(n, b) for n, b in budgets if n in heads]):
            c = np.array(heads[n]["contrastive"])
            lim = np.abs(c).max()
            im = ax.imshow(c, aspect="auto", cmap="coolwarm", vmin=-lim, vmax=lim)
            ax.set_title(f"{b} visual tokens", fontsize=9); ax.set_xlabel("head"); ax.set_ylabel("layer")
            fig.colorbar(im, ax=ax, fraction=.046)
        savefig(fig, fdir, "head_scores")
        names = [n for n, _ in budgets if n in heads]
        J = np.zeros((len(names), len(names)))
        for i, a in enumerate(names):
            for j, b_ in enumerate(names):
                A_ = {tuple(x) for x in heads[a]["top"][:20]}; B_ = {tuple(x) for x in heads[b_]["top"][:20]}
                J[i, j] = len(A_ & B_) / len(A_ | B_)
        fig, ax = plt.subplots(figsize=(3.6, 3.1))
        im = ax.imshow(J, vmin=0, vmax=1, cmap="viridis")
        labs = [n.replace("none", "").replace("hybrid", "") for n in names]
        ax.set_xticks(range(len(names))); ax.set_xticklabels(labs); ax.set_yticks(range(len(names))); ax.set_yticklabels(labs)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{J[i, j]:.2f}", ha="center", va="center", color="w" if J[i, j] < .6 else "k", fontsize=8)
        ax.set_title("Top-20 hallucination-head overlap (Jaccard)", fontsize=8)
        savefig(fig, fdir, "head_overlap")
        summary["head_overlap_jaccard"] = {"names": names, "J": J.tolist()}
        layer_hist = {n: np.bincount([l for l, _ in heads[n]["top"][:20]], minlength=len(heads[n]["contrastive"])).tolist()
                      for n in names}
        summary["hall_head_layers"] = layer_hist
        v = (heads.get("hybrid64") or {}).get("verify")
        if v:
            fig, ax = plt.subplots(figsize=(3.4, 3.1))
            ax.scatter(v["estimate"], v["exact"], s=14)
            ax.set_xlabel("attribution estimate"); ax.set_ylabel("exact zero-ablation effect")
            ax.set_title(f"Pearson r = {v['pearson']:.2f}", fontsize=9); ax.grid(alpha=.3)
            savefig(fig, fdir, "attribution_check")
            summary["attribution_check_pearson"] = v["pearson"]

    # ---------- TF-HH training curve
    log = load(os.path.join(res, "tfhh_hybrid64", "train_log.json"))
    if log:
        fig, ax = plt.subplots(figsize=(4.4, 3))
        st = [r["step"] for r in log]
        ax.plot(st, [r["ce"] for r in log], label="caption CE")
        ax2 = ax.twinx(); ax2.plot(st, [r["log_text_attn"] for r in log], color="#C44E52", label="log text-attn (hall. heads)")
        ax.set_xlabel("step"); ax.set_ylabel("CE"); ax2.set_ylabel("log text attention")
        fig.legend(fontsize=7, loc="upper right")
        savefig(fig, fdir, "tfhh_training")

    json.dump(summary, open(os.path.join(res, "summary.json"), "w"), indent=1)
    print("figures ->", fdir)


if __name__ == "__main__":
    main()
