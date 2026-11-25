"""
data.py — HotpotQA 数据加载与评测子集抽样（从零实现）

官方代码的抽样方式（hotpotqa.ipynb）：
    idxs = list(range(7405))
    random.Random(233).shuffle(idxs)
    idxs[:500]
即 seed=233 的随机 500 题。本次复现取前 n_questions 题（默认 100），
保持同一抽样序列，因此是官方 500 题子集的前缀，可与官方结果对照。

数据来源：HuggingFace `hotpotqa/hotpot_qa` 的 distractor 配置 validation 划分
（本机 huggingface.co 不可达，经 hf-mirror.com 镜像下载 parquet）。
"""
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

LOGGER = logging.getLogger(__name__)

HOTPO_QA_PARQUET = "hotpot_qa_distractor_validation.parquet"


def load_hotpotqa(path: Optional[Path] = None) -> pd.DataFrame:
    """载入 HotpotQA distractor validation 集。"""
    if path is None:
        from .configs import DATA_DIR
        path = DATA_DIR / HOTPO_QA_PARQUET
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"未找到 HotpotQA 数据文件：{path}\n"
            "请先运行：bash scripts/download_data.sh"
        )
    df = pd.read_parquet(path)
    LOGGER.info("HotpotQA distractor validation：%d 题", len(df))
    return df


def sample_subset(df: pd.DataFrame, n_questions: int, seed: int = 233) -> pd.DataFrame:
    """按官方方式抽样：random.Random(seed).shuffle(range(len(df))) 后取前 n 题。

    官方代码对 7405 条 dev 数据做同样操作，因此当本函数作用于同一数据集时，
    抽样序列与官方一致，取前 n 题即为官方 500 题子集的前缀。
    """
    idxs = list(range(len(df)))
    random.Random(seed).shuffle(idxs)
    idxs = idxs[:n_questions]
    return df.iloc[sorted(idxs)].reset_index(drop=True)


def build_records(df: pd.DataFrame) -> List[Dict[str, str]]:
    """转为统一记录格式。"""
    records: List[Dict[str, str]] = []
    for _, row in df.iterrows():
        records.append({
            "qid": str(row.get("id", "")),
            "question": str(row["question"]).strip(),
            "gold_answer": str(row["answer"]).strip(),
            "type": str(row.get("type", "")),
            "level": str(row.get("level", "")),
        })
    return records


def get_context_titles(row) -> List[str]:
    """取该题的 supporting facts 标题（用于诊断检索是否命中金标证据）。"""
    try:
        sf = row["supporting_facts"]
        return [str(t) for t in sf.get("title", [])]
    except Exception:  # noqa: BLE001
        return []
