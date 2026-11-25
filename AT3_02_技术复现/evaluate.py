"""
evaluate.py — 汇总结果、生成图表与对比表

职责：
  1. 读取 run_experiment.py 产出的 metrics_*.csv 与 per_sample_*.csv
  2. 生成结果图表（EM/F1 对比、完成率诊断等）
  3. 输出与原论文 Table 1 的逐项对照表
  4. 做配对显著性检验（McNemar / bootstrap），因为 100 题上小幅差异
     在统计上可能不显著，报告必须诚实呈现这一点

用法：
    python evaluate.py --tag main
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 显式设置中文字体（Windows 环境）：否则 matplotlib 回退到 DejaVu Sans 导致中文缺字
matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft JhengHei", "Microsoft YaHei", "SimHei", "DengXian", "SimSun",
]
matplotlib.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.configs import CONFIG, FIGURES_DIR, RESULTS_DIR

# 原论文 Table 1（PaLM-540B，HotpotQA EM / Fever Acc）
PAPER_TABLE1 = {
    "standard": {"hotpotqa": 28.7, "fever": 57.1},
    "cot":      {"hotpotqa": 29.4, "fever": 56.3},
    "cot_sc":   {"hotpotqa": 33.4, "fever": 60.4},
    "act":      {"hotpotqa": 25.7, "fever": 58.9},
    "react":    {"hotpotqa": 27.4, "fever": 60.9},
    "cot_sc_to_react": {"hotpotqa": 34.2, "fever": 64.6},
    "react_to_cot_sc": {"hotpotqa": 35.1, "fever": 62.0},
    "supervised_sota": {"hotpotqa": 67.5, "fever": 89.5},
}

METHOD_LABEL = {
    "standard": "Standard",
    "cot": "CoT",
    "act": "Act",
    "react": "ReAct",
}


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def wilson(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = (z / denom) * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def mcnemar_exact(a: List[int], b: List[int]) -> Dict[str, float]:
    """配对二分类的 McNemar 精确检验（binomial 版）。

    a、b 为两方法在同一批题上的 0/1 正确性向量。
    返回 b01/b10 与双尾 p 值。
    """
    b01 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    b10 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    n = b01 + b10
    if n == 0:
        return {"b01": b01, "b10": b10, "p": 1.0, "n_discordant": 0}
    # 精确二项检验（双尾）
    from math import comb
    k = min(b01, b10)
    p = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n) * 2
    return {"b01": b01, "b10": b10, "p": min(1.0, p), "n_discordant": n}


def bootstrap_diff(a: List[float], b: List[float], n_boot: int = 10000,
                   seed: int = 42) -> Dict[str, float]:
    """配對 bootstrap：估计 a-b 的 95% 置信区间。"""
    rng = random.Random(seed)
    n = len(a)
    diffs = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(float(np.mean([a[i] for i in idx]) - np.mean([b[i] for i in idx])))
    diffs.sort()
    lo = diffs[int(0.025 * n_boot)]
    hi = diffs[int(0.975 * n_boot)]
    return {"mean_diff": float(np.mean(a) - np.mean(b)),
            "ci95_lo": lo, "ci95_hi": hi}


def make_figures(metrics: List[Dict[str, str]], per_sample: List[Dict[str, str]]) -> List[Path]:
    """生成结果图表。"""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []

    methods = [m["method"] for m in metrics]
    labels = [METHOD_LABEL.get(m, m) for m in methods]
    em = [float(m["em"]) for m in metrics]
    f1 = [float(m["f1"]) for m in metrics]
    lo = [float(m["em_ci95_lo"]) for m in metrics]
    hi = [float(m["em_ci95_hi"]) for m in metrics]
    paper = [PAPER_TABLE1.get(m, {}).get("hotpotqa", np.nan) for m in methods]

    # --- 图 1：本次复现 EM vs 原论文 EM（双面板，各自可读）---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2))
    x = np.arange(len(methods))
    w = 0.38

    # 左：本次复现（带 95% CI）
    yerr = [np.array(em) - np.array(lo), np.array(hi) - np.array(em)]
    bars = ax1.bar(x, em, w, yerr=yerr, capsize=4, color="#4472C4", edgecolor="#2F5597")
    for b, v in zip(bars, em):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}",
                 ha="center", fontsize=9, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("HotpotQA Exact Match")
    ax1.set_title("本次复现（Qwen2.5-Coder-3B, 100 题）\n误差棒为 Wilson 95% CI")
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_ylim(0, max(em) * 1.45 + 0.05)

    # 右：原论文
    valid_paper = [(i, v) for i, v in enumerate(paper) if not np.isnan(v)]
    if valid_paper:
        idxs = [i for i, _ in valid_paper]
        vals = [v for _, v in valid_paper]
        bars2 = ax2.bar([x[i] for i in idxs], vals, w, color="#D9E2F3", edgecolor="#2F5597")
        for b, v in zip(bars2, vals):
            ax2.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.1f}",
                     ha="center", fontsize=9, fontweight="bold")
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels)
        ax2.set_ylim(0, max(vals) * 1.25)
    ax2.set_ylabel("HotpotQA Exact Match")
    ax2.set_title("原论文 Table 1（PaLM-540B）\n绝对值不可直接对齐")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("ReAct 复现：本次结果 vs 原论文（左侧为本次实测，右侧为原文数值）",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    p = FIGURES_DIR / "fig1_reproduction_vs_paper.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    paths.append(p)

    # --- 图 2：诊断指标（完成率 / 格式遵循率 / 检索命中率） ---
    fig, ax = plt.subplots(figsize=(9, 5.2))
    comp = [float(m["completion_rate"]) for m in metrics]
    fmt = [float(m["format_ok_rate"]) for m in metrics]
    srh = [float(m["search_hit_rate"]) for m in metrics]
    w = 0.26
    ax.bar(x - w, comp, w, label="完成率 (步数内产出 finish)", color="#4472C4")
    ax.bar(x, fmt, w, label="格式遵循率 (可解析出合法动作)", color="#70AD47")
    ax.bar(x + w, srh, w, label="检索命中率 (search 非空)", color="#ED7D31")
    for i in range(len(methods)):
        ax.text(i - w, comp[i] + 0.012, f"{comp[i]:.2f}", ha="center", fontsize=8)
        ax.text(i, fmt[i] + 0.012, f"{fmt[i]:.2f}", ha="center", fontsize=8)
        ax.text(i + w, srh[i] + 0.012, f"{srh[i]:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("比例")
    ax.set_title("诊断指标：小模型复现的行为特征\n(解释绝对分数与原论文差距的来源)")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.18)
    fig.tight_layout()
    p = FIGURES_DIR / "fig2_diagnostics.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    paths.append(p)

    # --- 图 3：Act vs ReAct 的配对逐题差异 ---
    if "act" in methods and "react" in methods:
        act_map = {r["qid"]: int(float(r["em"])) for r in per_sample if r["method"] == "act"}
        react_map = {r["qid"]: int(float(r["em"])) for r in per_sample if r["method"] == "react"}
        qids = sorted(set(act_map) & set(react_map))
        fig, ax = plt.subplots(figsize=(7.5, 5.2))
        categories = ["两者都对", "仅 ReAct 对", "仅 Act 对", "两者都错"]
        c = [sum(1 for q in qids if act_map[q] and react_map[q]),
             sum(1 for q in qids if not act_map[q] and react_map[q]),
             sum(1 for q in qids if act_map[q] and not react_map[q]),
             sum(1 for q in qids if not act_map[q] and not react_map[q])]
        colors = ["#BFBFBF", "#70AD47", "#ED7D31", "#D9D9D9"]
        bars = ax.bar(categories, c, color=colors, edgecolor="#404040")
        for b, v in zip(bars, c):
            ax.text(b.get_x() + b.get_width() / 2, v + 1, str(v), ha="center", fontweight="bold")
        ax.set_ylabel("题数")
        ax.set_title("Act vs ReAct 的逐题配对结果\n(检验论文「ReAct > Act」在 3B 模型上是否成立)")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        p = FIGURES_DIR / "fig3_act_vs_react_paired.png"
        fig.savefig(p, dpi=160)
        plt.close(fig)
        paths.append(p)

    # --- 图 4：相对排序对照（本次复现的核心验收标准）---
    pairs = [("cot", "standard"), ("act", "standard"), ("react", "act"), ("react", "cot")]
    have = [p for p in pairs if p[0] in methods and p[1] in methods]
    if have:
        fig, (axa, axb) = plt.subplots(1, 2, figsize=(12, 5.0))
        # 本次：相对 EM 差值（百分比点）
        deltas = []
        labels_p = []
        for a, b in have:
            ea = float(next(m["em"] for m in metrics if m["method"] == a))
            eb = float(next(m["em"] for m in metrics if m["method"] == b))
            deltas.append((ea - eb) * 100)
            labels_p.append(f"{METHOD_LABEL.get(a, a)}\n- {METHOD_LABEL.get(b, b)}")
        colors_a = ["#70AD47" if d > 0 else "#ED7D31" for d in deltas]
        bars = axa.bar(labels_p, deltas, color=colors_a, edgecolor="#404040")
        for b, d in zip(bars, deltas):
            axa.text(b.get_x() + b.get_width() / 2, d + (0.4 if d >= 0 else -0.9),
                     f"{d:+.1f}pp", ha="center", fontsize=9, fontweight="bold")
        axa.axhline(0, color="#404040", lw=1)
        axa.set_ylabel("ΔEM（百分點）")
        axa.set_title("本次复现的相对差值\n（綠=前者較高，橙=後者較高）")
        axa.grid(axis="y", alpha=0.3)

        # 原论文：同口径
        paper_deltas = []
        for a, b in have:
            pa = PAPER_TABLE1[a]["hotpotqa"]
            pb = PAPER_TABLE1[b]["hotpotqa"]
            paper_deltas.append(pa - pb)
        colors_b = ["#70AD47" if d > 0 else "#ED7D31" for d in paper_deltas]
        bars2 = axb.bar(labels_p, paper_deltas, color=colors_b, edgecolor="#404040")
        for b, d in zip(bars2, paper_deltas):
            axb.text(b.get_x() + b.get_width() / 2, d + (0.08 if d >= 0 else -0.28),
                     f"{d:+.1f}pp", ha="center", fontsize=9, fontweight="bold")
        axb.axhline(0, color="#404040", lw=1)
        axb.set_ylabel("ΔEM（百分點）")
        axb.set_title("原论文 Table 1 的相对差值\n（PaLM-540B）")
        axb.grid(axis="y", alpha=0.3)

        fig.suptitle("相对排序对照：本次复现 vs 原论文（本次為 100 題，差異均未達統計顯著）",
                     fontsize=12, fontweight="bold")
        fig.tight_layout()
        p = FIGURES_DIR / "fig4_relative_ordering.png"
        fig.savefig(p, dpi=160)
        plt.close(fig)
        paths.append(p)

    return paths


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()

    metrics_path = RESULTS_DIR / f"metrics_{args.tag}.csv"
    per_sample_path = RESULTS_DIR / f"per_sample_{args.tag}.csv"
    if not metrics_path.exists():
        print(f"找不到 {metrics_path}，请先运行 run_experiment.py")
        return 1

    metrics = read_csv(metrics_path)
    per_sample = read_csv(per_sample_path) if per_sample_path.exists() else []

    print("=" * 78)
    print("AT3 ② ReAct 复现 — 结果对照表")
    print("=" * 78)

    rows: List[Dict[str, object]] = []
    print(f"\n{'方法':<10}{'本次 EM':>10}{'95% CI':>18}{'本次 F1':>10}{'原文 EM':>10}{'差值':>9}")
    print("-" * 78)
    for m in metrics:
        meth = m["method"]
        e = float(m["em"])
        lo, hi = float(m["em_ci95_lo"]), float(m["em_ci95_hi"])
        f = float(m["f1"])
        pv = PAPER_TABLE1.get(meth, {}).get("hotpotqa")
        ps = f"{pv:.1f}" if pv is not None else "—"
        ds = f"{e - pv:.3f}" if pv is not None else "—"
        print(f"{METHOD_LABEL.get(meth, meth):<10}{e:>10.3f}"
              f"{f'[{lo:.3f}, {hi:.3f}]':>18}{f:>10.3f}{ps:>10}{ds:>9}")
        rows.append({"method": meth, "label": METHOD_LABEL.get(meth, meth),
                     "repro_em": e, "repro_em_ci95_lo": lo, "repro_em_ci95_hi": hi,
                     "repro_f1": f, "paper_em": pv, "diff": (e - pv) if pv is not None else None,
                     "completion_rate": float(m["completion_rate"]),
                     "format_ok_rate": float(m["format_ok_rate"]),
                     "avg_steps": float(m["avg_steps"]),
                     "search_hit_rate": float(m["search_hit_rate"])})

    # 配对显著性
    if per_sample:
        print("\n" + "=" * 78)
        print("配对显著性检验（100 题上小幅差异未必显著，务必如实报告）")
        print("=" * 78)
        em_map = {m: {r["qid"]: int(float(r["em"])) for r in per_sample if r["method"] == m}
                  for m in {r["method"] for r in per_sample}}
        pairs = [("react", "cot"), ("react", "act"), ("cot", "standard"), ("act", "standard")]
        for a, b in pairs:
            if a not in em_map or b not in em_map:
                continue
            qids = sorted(set(em_map[a]) & set(em_map[b]))
            if not qids:
                continue
            va = [em_map[a][q] for q in qids]
            vb = [em_map[b][q] for q in qids]
            mc = mcnemar_exact(va, vb)
            bs = bootstrap_diff([float(x) for x in va], [float(x) for x in vb])
            sig = "显著" if mc["p"] < 0.05 else "不显著"
            print(f"  {METHOD_LABEL.get(a, a):<8} vs {METHOD_LABEL.get(b, b):<8} "
                  f"ΔEM={bs['mean_diff']:+.3f} "
                  f"95%CI=[{bs['ci95_lo']:+.3f}, {bs['ci95_hi']:+.3f}]  "
                  f"McNemar p={mc['p']:.4f} ({sig})  "
                  f"仅{a[:1].upper()}对={mc['b01']} 仅{b[:1].upper()}对={mc['b10']}")
            rows.append({"comparison": f"{a}_vs_{b}", "delta_em": bs["mean_diff"],
                         "ci95_lo": bs["ci95_lo"], "ci95_hi": bs["ci95_hi"],
                         "mcnemar_p": mc["p"], "significant_0.05": mc["p"] < 0.05})

    # 图表
    paths = make_figures(metrics, per_sample)
    print("\n图表已生成：")
    for p in paths:
        print("  ", p)

    out = RESULTS_DIR / f"comparison_{args.tag}.csv"
    keys = sorted({k for r in rows for k in r})
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\n对照表已保存：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
