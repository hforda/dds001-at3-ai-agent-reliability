"""
evaluation.py — ReAct 的答案归一化与指标（从零实现）

归一化规则严格照抄官方 wrappers.py 的 normalize_answer，**顺序不可调换**：
    lower -> remove_punc -> remove_articles -> white_space_fix

论文正文与附录均未描述归一化规则（仅存在于官方代码），但 EM 的可比性完全依赖于此，
因此本次复现逐字实现。

官方代码 get_metrics() 中 pred / gt 变量名写反了（数值不受影响，因 EM/F1 对称），
本次实现修正了命名。

指标：
  - EM   ：归一化后完全相等（论文 Table 1 对 HotpotQA 只报 EM）
  - F1   ：SQuAD 式 token 重叠 F1（官方代码额外提供，本次一并输出）
  - 完成率、平均步数、格式遵循率、检索成功率（本次新增的诊断指标）
"""
from __future__ import annotations

import re
import string
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple


def normalize_answer(s: str) -> str:
    """官方 normalize_answer 的逐字实现。"""
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _squad_tokens(text: str) -> List[str]:
    """SQuAD 官方 tokenize：去标点、转小写、拆词。"""
    without_punc = "".join(ch for ch in text.lower() if ch not in set(string.punctuation))
    return without_punc.split()


def f1_score(prediction: str, ground_truth: str) -> Tuple[float, float, float]:
    """返回 (f1, precision, recall)。与官方 f1_score 一致。"""
    pred_tokens = _squad_tokens(prediction)
    gt_tokens = _squad_tokens(ground_truth)

    # 官方特殊规则：yes/no/noanswer 三者之一且不相等时直接判 0
    for tok in ("yes", "no", "noanswer"):
        if pred_tokens == [tok] or gt_tokens == [tok]:
            if pred_tokens != gt_tokens:
                return 0.0, 0.0, 0.0

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0, 0.0, 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1, precision, recall


def exact_match(prediction: str, ground_truth: str) -> bool:
    """归一化后完全相等。"""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def compute_metrics(prediction: str, ground_truth: str) -> Dict[str, float]:
    """单条样本的指标。"""
    em = float(exact_match(prediction, ground_truth))
    f1, precision, recall = f1_score(prediction, ground_truth)
    return {"em": em, "f1": f1, "precision": precision, "recall": recall}


def aggregate(per_sample: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """汇总一组样本的指标。"""
    if not per_sample:
        return {"em": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0, "n": 0}
    n = len(per_sample)
    return {
        "em": sum(s["em"] for s in per_sample) / n,
        "f1": sum(s["f1"] for s in per_sample) / n,
        "precision": sum(s["precision"] for s in per_sample) / n,
        "recall": sum(s["recall"] for s in per_sample) / n,
        "n": n,
    }


def wilson_interval(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """比例的 Wilson 95% 置信区间。

    100 题上的 EM 比例若直接比较，±1 个样本即 1 个百分点，统计功效有限。
    本函数用于在报告中给出区间，避免过度解读小幅差异。
    """
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = (z / denom) * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return (max(0.0, centre - margin), min(1.0, centre + margin))
