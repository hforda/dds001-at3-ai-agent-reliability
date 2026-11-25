"""
fill_report.py — 將實測結果自動填入 reproduction_report.md 的佔位符

避免手動抄寫數字造成錯誤 —— 所有佔位符的值都直接來自 results/ 下的實測 CSV。

用法：
    python fill_report.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.configs import RESULTS_DIR

REPORT = Path(__file__).resolve().parent / "docs" / "reproduction_report.md"


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pct(x: float) -> str:
    return f"{x * 100:.1f}"


def main() -> int:
    metrics = {m["method"]: m for m in read_csv(RESULTS_DIR / "metrics_main.csv")}
    comparison = read_csv(RESULTS_DIR / "comparison_main.csv")
    if not metrics:
        print("找不到 results/metrics_main.csv，請先運行 run_experiment.py")
        return 1

    subs: Dict[str, str] = {}
    for meth in ("standard", "cot", "act", "react"):
        m = metrics.get(meth)
        if not m:
            continue
        subs[f"RESULT_{meth.upper()}_EM"] = pct(float(m["em"]))
        subs[f"RESULT_{meth.upper()}_F1"] = pct(float(m["f1"]))
        subs[f"RESULT_{meth.upper()}_COMP"] = pct(float(m["completion_rate"]))
        subs[f"RESULT_{meth.upper()}_FMT"] = pct(float(m["format_ok_rate"]))
        subs[f"RESULT_{meth.upper()}_SRH"] = pct(float(m["search_hit_rate"]))

    # 相对排序（用原始数值比较方向）
    def em(m: str) -> float:
        return float(metrics[m]["em"])

    def cmp_block(a: str, b: str) -> tuple[str, str]:
        """返回 (差值描述, 是否一致的判定文字)。"""
        da, db = em(a), em(b)
        diff = (da - db) * 100
        if abs(diff) < 1e-9:
            return "兩者持平", "—"
        higher = a if da > db else b
        direction = f"{higher} 較高（{diff:+.1f}pp）"
        # 原文的方向
        paper = {"cot": 29.4, "standard": 28.7, "act": 25.7, "react": 27.4}
        p_diff = paper[a] - paper[b]
        p_higher = a if p_diff > 0 else b
        if p_diff == 0:
            match = "—（原文持平）"
        elif p_higher == higher:
            match = "✅ 一致"
        else:
            match = "❌ 不一致（方向相反）"
        return direction, match

    pairs = [("cot", "standard"), ("act", "standard"), ("react", "act"), ("react", "cot")]
    for a, b in pairs:
        if a in metrics and b in metrics:
            direction, match = cmp_block(a, b)
            subs[f"RESULT_{a.upper()}_VS_{b.upper()}"] = direction
            subs[f"RESULT_{a.upper()}_{b.upper()}_MATCH"] = match

    # McNemar p 值与 bootstrap CI
    for row in comparison:
        if "comparison" not in row:
            continue
        key = row["comparison"].upper()
        p_raw = (row.get("mcnemar_p") or "").strip()
        if not p_raw:
            continue
        subs[f"RESULT_{key}_P"] = f"{float(p_raw):.4f}"
        lo_raw = (row.get("ci95_lo") or "").strip()
        hi_raw = (row.get("ci95_hi") or "").strip()
        if lo_raw and hi_raw:
            subs[f"RESULT_{key}_CI"] = (f"[{float(lo_raw) * 100:+.1f}, "
                                        f"{float(hi_raw) * 100:+.1f}]pp")

    text = REPORT.read_text(encoding="utf-8")
    missing = []
    for k, v in subs.items():
        token = f"**{k}**"
        if token in text:
            text = text.replace(token, v)
        else:
            missing.append(k)
    REPORT.write_text(text, encoding="utf-8")

    print(f"已填入 {len(subs) - len(missing)} 個佔位符")
    if missing:
        print("未在報告中找到的佔位符：", missing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
