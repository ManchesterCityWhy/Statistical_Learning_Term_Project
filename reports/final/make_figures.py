"""Figures for the final report, drawn only from result files committed in this repository.

    python reports/final/make_figures.py        # writes reports/final/figures/*.pdf

Local ceilings are not stored in the round-2 CSVs, so they are recovered from the
(raw metric, local score) pairs: the calibrated score is an exact function of the four raw
metrics and the four anchors, and 93 development rows pin the anchors down to ~1e-13.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
R2 = ROOT / "experiments" / "round2_seed42"
OUT = Path(__file__).resolve().parent / "figures"
METRICS = {"de_score": (.25, True), "de_direction": (.25, True), "mmd_u": (.30, False), "variogram": (.20, False)}
LABELS = {"de_score": "DES 差异基因", "de_direction": "DCS 变化方向", "mmd_u": "MMD 细胞分布", "variogram": "CSS 基因共变"}
# Reference categorical slots 1-4 (validated adjacent order), text stays in ink tokens.
COLORS = {"de_score": "#2a78d6", "de_direction": "#eb6834", "mmd_u": "#1baf7a", "variogram": "#eda100"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"

plt.rcParams.update({
    "font.family": ["Hiragino Sans GB", "STHeiti", "sans-serif"], "pdf.fonttype": 42,
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.unicode_minus": False,
})


def skill(value, floor, ceiling, higher):
    distance = ceiling - value if higher else value - ceiling
    return 1.0 if distance <= 0 else abs(floor - ceiling) / (abs(floor - ceiling) + distance)


def score(row, floor, ceiling):
    return 100 * sum(w * skill(row[k], floor[k], ceiling[k], hi) for k, (w, hi) in METRICS.items())


def recover_ceiling(frame, floor):
    keys = list(METRICS)

    def residual(x):
        ceiling = dict(zip(keys, x))
        return [score(r, floor, ceiling) - r["score"] for _, r in frame.iterrows()]

    fit = least_squares(residual, [.85, .79, .004, .00016], x_scale=[.1, .1, .001, .0001])
    assert np.abs(fit.fun).max() < 1e-6, "anchors not identified"
    return dict(zip(keys, fit.x))


def load(split):
    frames = []
    for run in ("expanded_seed42", "composition_seed42", "composition_extended_seed42", "composition_refined_seed42"):
        frame = pd.read_csv(R2 / run / f"{split}.csv")
        frames.append(frame.assign(run=run)[["run", "name", *METRICS, "score"]])
    legacy = pd.read_csv(R2 / "expanded_seed42" / "legacy_matched.csv")
    legacy = legacy[legacy["split"] == split].rename(columns={"candidate": "name"})
    frames.append(legacy.assign(run="legacy")[["run", "name", *METRICS, "score"]])
    return pd.concat(frames, ignore_index=True)


def contributions(row, floor, ceiling):
    return {k: 100 * w * skill(row[k], floor[k], ceiling[k], hi) for k, (w, hi) in METRICS.items()}


def figure_contributions(dev, floor, ceiling):
    expanded = dev[dev["run"] == "expanded_seed42"]
    rows = [
        ("复制最近阶段（下限）", dev[dev["name"] == "copy_last"].iloc[0]),
        ("旧方法：类型平移+bulk", dev[dev["name"] == "legacy_bulk_type_075"].iloc[0]),
        ("最佳类型外推（36 组）", expanded[expanded["name"] != "copy_last"].sort_values("score").iloc[-1]),
        ("组成反卷积+bulk（选定）", dev[dev["name"] == "deconv3_mix1_bulk3"].iloc[0]),
    ][::-1]
    fig, ax = plt.subplots(figsize=(6.3, 2.45))
    for y, (label, row) in enumerate(rows):
        left = 0.0
        parts = contributions(row, floor, ceiling)
        for key, value in parts.items():
            ax.barh(y, value - .35, left=left, height=.62, color=COLORS[key], linewidth=0, zorder=2)
            if value > 7:
                ax.text(left + (value - .35) / 2, y, f"{value:.1f}", ha="center", va="center",
                        color="white" if key in ("de_score", "de_direction") else INK, fontsize=7.5, zorder=3)
            left += value
        ax.text(left + 1, y, f"{left:.1f}", va="center", color=INK, fontsize=8.5, fontweight="bold")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=8.5)
    ax.axvline(50, color=INK2, lw=.8, ls=(0, (3, 2)), zorder=0)
    ax.text(50, len(rows) - .45, "下限 50", ha="center", va="bottom", color=INK2, fontsize=7.5)
    ax.set_xlim(0, 100)
    ax.set_xlabel("本地校准分数（每段 = 权重 × 该指标 skill × 100；满分 100）")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[k]) for k in METRICS]
    ax.legend(handles, [LABELS[k] for k in METRICS], ncol=4, frameon=False, fontsize=7.5,
              loc="lower center", bbox_to_anchor=(.42, 1.06), handlelength=1, columnspacing=1.2)
    fig.tight_layout()
    fig.savefig(OUT / "fig_contributions.pdf")
    plt.close(fig)
    return {label: {**contributions(row, floor, ceiling), "score": row["score"]} for label, row in rows}


def parse(name):
    """'deconv3_mix1_bulk1.5' -> (3.0, 1.0, 1.5)"""
    strength, mix, bulk = name.split("_")
    return float(strength[6:]), float(mix[3:]), float(bulk[4:])


def figure_sensitivity(dev):
    comp = dev[dev["name"].str.startswith("deconv")].copy()
    comp[["s", "mix", "bulk"]] = comp["name"].apply(lambda n: pd.Series(parse(n)))
    comp = comp[comp["mix"] == 1].groupby(["s", "bulk"], as_index=False)["score"].mean()
    fig, (left, right) = plt.subplots(1, 2, figsize=(6.3, 2.5), gridspec_kw={"width_ratios": [1.35, 1]})
    styles = {0.0: ("#2a78d6", "不加表达迁移"), 1.5: ("#eb6834", "表达迁移 1.5"), 5.0: ("#1baf7a", "表达迁移 5")}
    for bulk, (color, label) in styles.items():
        sub = comp[comp["bulk"] == bulk].sort_values("s")
        left.plot(sub["s"], sub["score"], color=color, lw=2, marker="o", ms=4.5,
                  markeredgecolor="white", markeredgewidth=1.2, label=label)
    left.set_xscale("log", base=2)
    left.set_xticks([.25, .5, 1, 2, 3, 5], ["0.25", "0.5", "1", "2", "3", "5"])
    left.set_xlabel("组成趋势强度 s（混合比例 1）")
    left.set_ylabel("选模集分数")
    left.axhline(50, color=INK2, lw=.8, ls=(0, (3, 2)))
    left.legend(frameon=False, fontsize=7.5, loc="upper left", numpoints=1, handlelength=1.6)
    left.grid(axis="y", color=GRID, lw=.6)
    left.set_title("a  组成调整的强度", loc="left", fontsize=9, color=INK)
    s3 = comp[comp["s"] == 3].sort_values("bulk")
    right.plot(s3["bulk"], s3["score"], color="#6250d6", lw=2, marker="o", ms=4.5,
               markeredgecolor="white", markeredgewidth=1.2)
    for _, r in s3.iterrows():
        if r["bulk"] in (3, 20):
            right.annotate(f"{r['score']:.1f}", (r["bulk"], r["score"]), textcoords="offset points",
                           xytext=(-6, -14) if r["bulk"] == 3 else (-30, 2), fontsize=7.5, color=INK)
    right.axhline(50, color=INK2, lw=.8, ls=(0, (3, 2)))
    right.set_xlim(-1, 21.5)
    right.set_xlabel("细胞内表达迁移强度（s = 3）")
    right.grid(axis="y", color=GRID, lw=.6)
    right.set_title("b  表达迁移的强度", loc="left", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "fig_sensitivity.pdf")
    plt.close(fig)


def pretty(name):
    if name == "copy_last":
        return "复制最近阶段"
    if name == "legacy_bulk_type_075":
        return "旧方法 bulk_type_075"
    if name.startswith("deconv"):
        s, _, bulk = parse(name)
        return f"组成反卷积 s={s:g}，表达 {bulk:g}"
    kind, typ, bulk = name.split("_")
    word = "加法" if kind == "add" else "比例"
    return f"类型{word}外推 ×{typ[4:]}，表达 {bulk[4:]}"


def figure_selection(dev, confirm):
    pairs = confirm.merge(dev, on="name", suffixes=("_confirm", "_dev"))
    pairs = pairs.drop_duplicates("name").sort_values("score_dev")
    fig, ax = plt.subplots(figsize=(6.3, 3.5))
    for y, (_, r) in enumerate(pairs.iterrows()):
        ax.plot([r["score_dev"], r["score_confirm"]], [y, y], color=GRID, lw=2.2, zorder=1)
        ax.scatter(r["score_dev"], y, s=30, color="#2a78d6", edgecolor="white", linewidth=1.2, zorder=3)
        ax.scatter(r["score_confirm"], y, s=30, color="#eb6834", edgecolor="white", linewidth=1.2, zorder=3)
    ax.set_yticks(range(len(pairs)), [pretty(n) for n in pairs["name"]], fontsize=7.5)
    audits = [59.27223131003188, 60.153809244229414, 60.48178365231743]
    top = list(pairs["name"]).index("deconv3_mix1_bulk3")
    ax.scatter(audits, [top + .32] * 3, s=14, marker="D", color="#1baf7a", edgecolor="white", linewidth=.8, zorder=3)
    ax.set_xlabel("本地校准分数")
    ax.grid(axis="x", color=GRID, lw=.6)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    handles = [ax.scatter([], [], s=30, color=c, edgecolor="white") for c in ("#2a78d6", "#eb6834")]
    handles.append(ax.scatter([], [], s=14, marker="D", color="#1baf7a"))
    ax.legend(handles, ["选模集（参与挑选）", "复核集（不参与挑选）", "三组审计集"], frameon=False, fontsize=7.5,
              ncol=3, loc="lower center", bbox_to_anchor=(.45, 1.0))
    fig.tight_layout()
    fig.savefig(OUT / "fig_selection.pdf")
    plt.close(fig)
    return pairs[["name", "score_dev", "score_confirm"]]


def main():
    OUT.mkdir(exist_ok=True)
    dev, confirm = load("development"), load("confirmation")
    floors = {s: d[d["name"] == "copy_last"].iloc[0] for s, d in (("development", dev), ("confirmation", confirm))}
    dev_ceiling = recover_ceiling(dev, floors["development"])
    confirm_ceiling = recover_ceiling(confirm.drop_duplicates("name"), floors["confirmation"])
    print("development anchors", {k: (float(floors["development"][k]), dev_ceiling[k]) for k in METRICS})
    print("confirmation anchors", {k: (float(floors["confirmation"][k]), confirm_ceiling[k]) for k in METRICS})
    print(pd.DataFrame(figure_contributions(dev, floors["development"], dev_ceiling)).T.round(2))
    figure_sensitivity(dev)
    print(figure_selection(dev, confirm).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
