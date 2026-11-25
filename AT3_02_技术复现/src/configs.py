"""
configs.py — AT3 ② ReAct 复现的集中参数管理

所有可调参数集中于此，路径均可通过环境变量或本文件覆盖。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict

# ---------------------------------------------------------------------------
# 路径：全部相对于本文件的上级目录（工程根），保证工程可整体搬移
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
LOGS_DIR = RESULTS_DIR / "logs"
TRAJECTORIES_DIR = RESULTS_DIR / "trajectories"
CACHE_DIR = DATA_DIR / "cache"

# 环境变量可覆盖模型与数据路径
MODEL_PATH = os.environ.get("REACT_MODEL_PATH", str(MODELS_DIR / "Qwen2.5-Coder-3B-Instruct"))
WIKIPEDIA_CACHE = Path(os.environ.get("REACT_WIKI_CACHE", str(CACHE_DIR / "wikipedia")))


@dataclass
class ReActConfig:
    """ReAct 复现的全部配置项。"""

    # ---------------- 模型 ----------------
    model_path: str = MODEL_PATH
    dtype: str = "bfloat16"          # RTX 3070 支持 bf16
    device: str = "cuda" if os.environ.get("REACT_FORCE_CPU", "0") != "1" else "cpu"
    max_new_tokens: int = 96        # 官方 GPT-3 配置为 100；小模型需略放宽以容纳完整动作
    temperature: float = 0.0         # 官方 text-davinci-002 为贪心解码 temperature=0
    top_p: float = 1.0
    # 批量生成：显著加速多题评测
    batch_size: int = 4

    # ---------------- 任务与评测 ----------------
    task: str = "hotpotqa"           # 本次仅复现 HotpotQA
    n_questions: int = 100           # 原论文官方代码为 500 题（seed=233）；本次缩至 100
    sample_seed: int = 233           # 与官方代码一致，便于对照
    max_steps: int = 7               # 论文 §3.2：HotpotQA 设 7 步
    # 步数用尽时的兜底：论文未规定单跑 ReAct 的兜底，官方代码用 finish[]（空答案）
    # True = 严格照官方代码；False = 允许一次「无答案」的额外作答尝试
    strict_step_exhaustion: bool = True

    # ---------------- 方法选择 ----------------
    methods: tuple = ("standard", "cot", "act", "react")
    # 是否启用 ReAct→CoT-SC 回退（本轮不启用 CoT-SC，见 paper_analysis 范围界定）
    enable_cot_sc: bool = False
    cot_sc_samples: int = 21         # 论文 §3.2 规定
    cot_sc_temperature: float = 0.7

    # ---------------- 环境 ----------------
    wiki_cache_dir: str = str(WIKIPEDIA_CACHE)
    wiki_request_timeout: float = 12.0
    wiki_max_retries: int = 2
    wiki_user_agent: str = (
        "AT3-ReAct-Reproduction/1.0 (coursework; contact: student@example.edu)"
    )
    # search 返回的句子数（论文 §3.1：first 5 sentences）
    wiki_search_sentences: int = 5

    # ---------------- 运行控制 ----------------
    seed: int = 42
    save_trajectories: bool = True
    # 单条轨迹生成失败的最大重试
    max_generation_retries: int = 2

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def ensure_dirs(self) -> None:
        for d in (DATA_DIR, RESULTS_DIR, FIGURES_DIR, LOGS_DIR, TRAJECTORIES_DIR, CACHE_DIR):
            d.mkdir(parents=True, exist_ok=True)


CONFIG = ReActConfig()
