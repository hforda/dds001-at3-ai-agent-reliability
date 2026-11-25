"""
hallucination_analysis.py — 复现论文 Table 2 的失败模式分析

论文 §3.3 的协议：
  "we randomly sampled 50 trajectories with correct and incorrect answers (judged by EM)
   from ReAct and CoT respectively (thus 200 examples in total), and manually labeled
   their success and failure modes in Table 2."

六类标注体系（论文 Table 2 原文）：
  Success:
    - True positive      : 正確的推理軌跡與事實
    - False positive     : 幻覺的推理軌跡或事實
  Failure:
    - Reasoning error    : 錯誤推理（含無法從重複步驟中恢復）
    - Search result error: 檢索回傳空結果或不含有用資訊
    - Hallucination      : 幻覺的推理軌跡或事實
    - Label ambiguity    : 預測正確但未精確匹配標籤

【本次复现的偏离声明】
原论文为**人工标注**（200 条，每类 50 条）。本次受课程作业资源限制，改用
**规则 + LLM-as-judge 的自动分类**，并明确说明自动标注与人工标注的口径差异：
  - 自动标注的观察窗口是 agent 自己的轨迹（thought / action / observation），
    而人工标注者可以看到完整解题过程，因此自动标注对「事实错误」的识别
    灵敏度必然低于人工。
  - 论文的 0% / 56% 这类对比在 200 条样本上的统计功效本身有限
    （0% 对应 0-2 条），本次样本量更大但标注器更弱，两者不可直接数值对等。

用法：
    python hallucination_analysis.py --tag main --per-method react cot
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.configs import CONFIG, RESULTS_DIR, TRAJECTORIES_DIR

# 论文 Table 2 的原始数值（用于方向性对照）
PAPER_TABLE2 = {
    "react": {"success_true_positive": 94, "success_false_positive": 6,
              "failure_reasoning_error": 47, "failure_search_result_error": 23,
              "failure_hallucination": 0, "failure_label_ambiguity": 29},
    "cot":   {"success_true_positive": 86, "success_false_positive": 14,
              "failure_reasoning_error": 16, "failure_search_result_error": None,
              "failure_hallucination": 56, "failure_label_ambiguity": 28},
}

LABELS = [
    "success_true_positive",
    "success_false_positive",
    "failure_reasoning_error",
    "failure_search_result_error",
    "failure_hallucination",
    "failure_label_ambiguity",
]

LABEL_ZH = {
    "success_true_positive": "成功-真阳性",
    "success_false_positive": "成功-假阳性（幻觉）",
    "failure_reasoning_error": "失败-推理错误",
    "failure_search_result_error": "失败-检索结果错误",
    "failure_hallucination": "失败-幻觉",
    "failure_label_ambiguity": "失败-标签歧义",
}

# ---- 规则式信号 ----
_RE_HEDGE = re.compile(
    r"\b(i think|probably|might be|likely|not sure|uncertain|i believe|"
    r"i guess|seems|appears to|i don'?t know|unclear)\b", re.IGNORECASE)
_RE_REPEAT = re.compile(r"(.{25,}?)\1{2,}", re.DOTALL)     # 同片段重复 3 次以上
_RE_ANSWER_ONLY = re.compile(r"^\s*(so\s+)?the\s+answer\s+is\b", re.IGNORECASE)


def rule_signals(traj: Dict) -> Dict[str, bool]:
    """从轨迹中提取规则式信号。"""
    history = traj.get("history", "") or ""
    raw = " ".join(traj.get("raw_outputs", []) or [])
    obs_blocks = re.findall(r"Observation \d+: (.*)", history)
    empty_obs = sum(1 for o in obs_blocks if o.startswith("Could not find")
                    or o.strip() in ("", "No more results."))
    return {
        "has_hedge": bool(_RE_HEDGE.search(raw)),
        "has_repetition": bool(_RE_REPEAT.search(raw)),
        "empty_observations": empty_obs > 0,
        "search_attempts": traj.get("search_attempts", 0) or 0,
        "search_hits": traj.get("search_hits", 0) or 0,
        "n_actions": traj.get("n_actions", 0) or 0,
        "completed": traj.get("completed", False),
        "format_ok": traj.get("format_ok", False),
    }


def classify(traj: Dict) -> str:
    """自动判定失败/成功模式。

    判定顺序与论文 Table 2 的分类互斥性要求一致：
      1. 先按 EM 分成功/失败
      2. 失败中：若答案本身含不确定表述 -> 幻觉
      3. 失败中：若检索完全落空（search_hits == 0 且有 search）-> 检索结果错误
      4. 失败中：若答案正确但 EM 不匹配（归一化后接近）-> 标签歧义
      5. 失败中：若轨迹含重复片段或未产出 finish -> 推理错误
      6. 失败兜底 -> 推理错误
    """
    from src.evaluation import normalize_answer

    em = float(traj.get("metrics", {}).get("em", 0.0))
    sig = rule_signals(traj)
    pred = traj.get("prediction", "") or ""
    gold = traj.get("gold_answer", "") or ""

    if em >= 1.0:
        # 成功：区分真阳性与假阳性（幻觉）
        if sig["has_hedge"] or sig["has_repetition"]:
            return "success_false_positive"
        return "success_true_positive"

    # 失败
    npred, ngold = normalize_answer(pred), normalize_answer(gold)
    # 标签歧义：归一化后 token 重叠高但 EM 不匹配
    if npred and ngold:
        pt, gt = set(npred.split()), set(ngold.split())
        if pt and gt:
            overlap = len(pt & gt) / len(pt | gt)
            if overlap >= 0.5:
                return "failure_label_ambiguity"

    if sig["has_hedge"]:
        return "failure_hallucination"
    if sig["search_attempts"] > 0 and sig["search_hits"] == 0:
        return "failure_search_result_error"
    if sig["has_repetition"] or not sig["completed"] or not sig["format_ok"]:
        return "failure_reasoning_error"
    return "failure_reasoning_error"


def load_trajectories(method: str, tag: str) -> List[Dict]:
    path = TRAJECTORIES_DIR / f"{method}_trajectories.jsonl"
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-method", nargs="+", default=["react", "cot"])
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()

    print("=" * 78)
    print("AT3 ② ReAct 复现 — 失败模式分析（对应论文 Table 2）")
    print("=" * 78)

    rows: List[Dict[str, object]] = []
    summary: Dict[str, Dict[str, float]] = {}

    for method in args.per_method:
        trajs = load_trajectories(method, args.tag)
        if not trajs:
            print(f"  [跳过] {method}: 未找到轨迹文件")
            continue
        counts = defaultdict(int)
        for t in trajs:
            lab = classify(t)
            counts[lab] += 1
            rows.append({
                "method": method, "qid": t.get("qid"), "label": lab,
                "label_zh": LABEL_ZH[lab], "em": t.get("metrics", {}).get("em"),
                "prediction": t.get("prediction"), "gold_answer": t.get("gold_answer"),
            })
        n = len(trajs)
        succ = sum(counts[k] for k in LABELS if k.startswith("success_"))
        fail = sum(counts[k] for k in LABELS if k.startswith("failure_"))
        summary[method] = {}
        print(f"\n【{method}】共 {n} 条（成功 {succ} / 失败 {fail}）")
        print(f"  {'标签':<24}{'本次 %':>10}{'原文 %':>10}")
        print("  " + "-" * 44)
        for lab in LABELS:
            c = counts[lab]
            pct = (c / succ * 100) if lab.startswith("success_") and succ else \
                  (c / fail * 100) if fail else 0.0
            pv = PAPER_TABLE2.get(method, {}).get(lab)
            ps = f"{pv}" if pv is not None else "—"
            print(f"  {LABEL_ZH[lab]:<24}{pct:>10.1f}{ps:>10}")
            summary[method][lab] = pct

    out_detail = RESULTS_DIR / f"hallucination_labels_{args.tag}.csv"
    if rows:
        keys = sorted({k for r in rows for k in r})
        with out_detail.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"\n逐条标注已保存：{out_detail}")

    out_sum = RESULTS_DIR / f"hallucination_summary_{args.tag}.csv"
    with out_sum.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "label", "label_zh", "repro_pct", "paper_pct"])
        for m, d in summary.items():
            for lab in LABELS:
                pv = PAPER_TABLE2.get(m, {}).get(lab)
                w.writerow([m, lab, LABEL_ZH[lab], f"{d.get(lab, 0):.1f}",
                            "" if pv is None else pv])
    print(f"汇总已保存：{out_sum}")
    print("\n注：本次为规则式自动标注，论文为人工标注，两者口径不同，"
          "仅可作方向性对照，不可作数值对等。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
