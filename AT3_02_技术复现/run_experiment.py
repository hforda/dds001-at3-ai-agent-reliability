"""
run_experiment.py — AT3 ② ReAct 复现的主入口

对 HotpotQA 的 4 条 prompting baseline（Standard / CoT / Act / ReAct）运行评测，
输出逐题轨迹、汇总指标与诊断统计。

用法：
    python run_experiment.py --methods standard cot act react --n-questions 100
    python run_experiment.py --methods react --n-questions 20     # 快速冒烟
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.agent import ReActAgent, Trajectory
from src.batch_runner import BatchReActRunner
from src.configs import CONFIG, LOGS_DIR, RESULTS_DIR, TRAJECTORIES_DIR, ReActConfig
from src.data import build_records, load_hotpotqa, sample_subset
from src.evaluation import aggregate, wilson_interval
from src.llm import LocalLLM
from src.prompts import build_few_shot
from src.wikipedia_env import WikipediaEnv

LOGGER = logging.getLogger("react_repro")


def set_seed(seed: int) -> None:
    """固定所有随机源，保证实验可重跑。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ReAct 复现实验")
    p.add_argument("--methods", nargs="+", default=list(CONFIG.methods),
                   choices=["standard", "cot", "act", "react"],
                   help="要评测的 prompting 方法")
    p.add_argument("--n-questions", type=int, default=CONFIG.n_questions,
                   help="评测题数（官方为 500，本次默认 100）")
    p.add_argument("--max-steps", type=int, default=CONFIG.max_steps,
                   help="ReAct/Act 的步数上限（论文 HotpotQA 为 7）")
    p.add_argument("--model-path", type=str, default=CONFIG.model_path)
    p.add_argument("--device", type=str, default=CONFIG.device)
    p.add_argument("--dtype", type=str, default=CONFIG.dtype)
    p.add_argument("--batch-size", type=int, default=CONFIG.batch_size)
    p.add_argument("--max-new-tokens", type=int, default=CONFIG.max_new_tokens)
    p.add_argument("--seed", type=int, default=CONFIG.seed)
    p.add_argument("--sample-seed", type=int, default=CONFIG.sample_seed,
                   help="HotpotQA 子集抽样种子（官方为 233）")
    p.add_argument("--tag", type=str, default="main",
                   help="本次运行的结果标签（用于区分多次运行）")
    p.add_argument("--limit-interactive", action="store_true",
                   help="仅跑前 N 题（配合 --n-questions 使用）")
    return p.parse_args()


def write_trajectories(trajs: List[Trajectory], method: str, out_dir: Path) -> Path:
    """把逐题轨迹写成 JSONL，便于复核。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{method}_trajectories.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for t in trajs:
            rec = {
                "qid": t.qid, "question": t.question, "gold_answer": t.gold_answer,
                "method": t.method, "prediction": t.prediction,
                "steps": t.steps, "n_actions": t.n_actions,
                "n_invalid_actions": t.n_invalid_actions, "n_badcalls": t.n_badcalls,
                "completed": t.completed, "format_ok": t.format_ok,
                "search_attempts": t.search_attempts, "search_hits": t.search_hits,
                "lookups": t.lookups, "metrics": t.metrics,
                "error": t.error, "history": t.history,
                "raw_outputs": t.raw_outputs,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path


def summarize(trajs: List[Trajectory], method: str) -> Dict[str, float]:
    """汇总某方法的指标与诊断统计。"""
    agg = aggregate([t.metrics for t in trajs])
    n = len(trajs)
    em_hits = sum(t.metrics["em"] for t in trajs)
    lo, hi = wilson_interval(int(em_hits), n)
    total_search = sum(t.search_attempts for t in trajs)
    total_hits = sum(t.search_hits for t in trajs)
    return {
        "method": method,
        "n": n,
        "em": agg["em"],
        "em_ci95_lo": lo,
        "em_ci95_hi": hi,
        "f1": agg["f1"],
        "precision": agg["precision"],
        "recall": agg["recall"],
        "completion_rate": sum(t.completed for t in trajs) / n if n else 0.0,
        "format_ok_rate": sum(t.format_ok for t in trajs) / n if n else 0.0,
        "avg_steps": sum(t.steps for t in trajs) / n if n else 0.0,
        "avg_actions": sum(t.n_actions for t in trajs) / n if n else 0.0,
        "avg_invalid_actions": sum(t.n_invalid_actions for t in trajs) / n if n else 0.0,
        "avg_badcalls": sum(t.n_badcalls for t in trajs) / n if n else 0.0,
        "avg_repeated_actions": sum(t.n_repeated_actions for t in trajs) / n if n else 0.0,
        "loop_rate": (sum(1 for t in trajs if t.n_repeated_actions > 0) / n) if n else 0.0,
        "search_hit_rate": (total_hits / total_search) if total_search else 0.0,
        "total_search": total_search,
        "total_lookups": sum(t.lookups for t in trajs),
    }


def main() -> int:
    args = parse_args()
    CONFIG.ensure_dirs()          # 必须在 logging 初始化前创建结果目录
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(LOGS_DIR / f"run_{args.tag}.log", encoding="utf-8")],
    )
    set_seed(args.seed)

    LOGGER.info("=" * 72)
    LOGGER.info("AT3 ② ReAct 复现实验")
    LOGGER.info("模型：%s", args.model_path)
    LOGGER.info("方法：%s", ", ".join(args.methods))
    LOGGER.info("题数：%d | 最大步数：%d | seed：%d", args.n_questions, args.max_steps, args.seed)
    LOGGER.info("=" * 72)

    # ---- 数据 ----
    df = load_hotpotqa()
    sub = sample_subset(df, args.n_questions, seed=args.sample_seed)
    records = build_records(sub)
    LOGGER.info("抽样子集：%d 题（抽样种子 %d，与官方 500 题序列一致）",
                len(records), args.sample_seed)

    # ---- 环境 ----
    env = WikipediaEnv(
        cache_dir=CONFIG.wiki_cache_dir,
        timeout=CONFIG.wiki_request_timeout,
        max_retries=CONFIG.wiki_max_retries,
        user_agent=CONFIG.wiki_user_agent,
        search_sentences=CONFIG.wiki_search_sentences,
    )

    # ---- 模型 ----
    llm = LocalLLM(
        model_path=args.model_path,
        dtype=args.dtype,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=CONFIG.temperature,
        batch_size=args.batch_size,
    )
    runner = BatchReActRunner(
        llm=llm, env=env, max_steps=args.max_steps,
        max_generation_retries=CONFIG.max_generation_retries,
        batch_size=args.batch_size,
    )

    all_summaries: List[Dict[str, float]] = []
    all_rows: List[Dict[str, object]] = []

    for method in args.methods:
        LOGGER.info("-" * 72)
        LOGGER.info("开始方法：%s", method)
        t0 = time.time()
        # 批量执行：所有题目同步推进同一轮，GPU 利用率显著高于逐题串行
        trajs = runner.run(method, records)
        elapsed = time.time() - t0

        em = sum(t.metrics["em"] for t in trajs) / len(trajs) if trajs else 0.0
        LOGGER.info("  [%s] %d 题完成，累计 EM=%.3f 耗时=%.0fs", method, len(trajs), em, elapsed)

        summary = summarize(trajs, method)
        summary["wall_clock_sec"] = elapsed
        all_summaries.append(summary)
        LOGGER.info("方法 %s 完成：EM=%.3f [%.3f, %.3f]  F1=%.3f  完成率=%.3f  耗时=%.0fs",
                    method, summary["em"], summary["em_ci95_lo"], summary["em_ci95_hi"],
                    summary["f1"], summary["completion_rate"], elapsed)

        if CONFIG.save_trajectories:
            p = write_trajectories(trajs, method, TRAJECTORIES_DIR)
            LOGGER.info("轨迹已保存：%s", p)

        for t in trajs:
            all_rows.append({
                "method": method, "qid": t.qid, "question": t.question,
                "gold_answer": t.gold_answer, "prediction": t.prediction,
                "em": t.metrics["em"], "f1": t.metrics["f1"],
                "steps": t.steps, "n_actions": t.n_actions,
                "n_invalid_actions": t.n_invalid_actions, "n_badcalls": t.n_badcalls,
                "n_repeated_actions": t.n_repeated_actions,
                "completed": t.completed, "format_ok": t.format_ok,
                "search_attempts": t.search_attempts, "search_hits": t.search_hits,
                "lookups": t.lookups, "error": t.error,
            })

    # ---- 输出 ----
    metrics_path = RESULTS_DIR / f"metrics_{args.tag}.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_summaries[0].keys()))
        writer.writeheader()
        writer.writerows(all_summaries)
    LOGGER.info("汇总指标已保存：%s", metrics_path)

    per_sample_path = RESULTS_DIR / f"per_sample_{args.tag}.csv"
    with per_sample_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    LOGGER.info("逐题结果已保存：%s", per_sample_path)

    LOGGER.info("=" * 72)
    LOGGER.info("汇总：")
    for s in all_summaries:
        LOGGER.info("  %-9s EM=%.3f  F1=%.3f  完成率=%.3f  格式遵循率=%.3f  平均步数=%.2f  检索命中率=%.3f",
                    s["method"], s["em"], s["f1"], s["completion_rate"],
                    s["format_ok_rate"], s["avg_steps"], s["search_hit_rate"])
    LOGGER.info("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
