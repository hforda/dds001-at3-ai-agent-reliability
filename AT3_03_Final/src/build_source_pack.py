"""
build_source_pack.py — 由原始 arXiv 分片构建本地离线来源包

来源数据集：RealTimeData/arxiv_alltime（arXiv 论文元数据 + 正文）
  - 选取 2023-01 / 2023-02 / 2023-03 三个月份分片
  - 字段：entry_id / published / title / authors / primary_category / categories / text

产出（source_pack/ 下，全部离线可读）：
  documents.jsonl   每行一篇文档：doc_id, title, authors, categories, published, url, sentences[]
  index.json        来源包元数据：数据集信息、许可、SHA256、文档数、类别分布
  sentences/        （可选）分片存储——本实现直接内嵌于 documents.jsonl 的 sentences 字段

设计要点：
  1. doc_id 采用 arXiv entry_id 的稳定部分（如 2301.11659v2），可回溯到 arxiv.org 页面。
  2. 正文切分为**带全局句号的句子列表**，每句带本地序号 s_idx，
     使 citation 能精确定位到「文档 ID + 标题 + 第 n 句」，并可逐字回查。
  3. 全部 SHA256 在本脚本计算并写入 index.json，供 provenance 引用。
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent   # 工程根目录
RAW = ROOT / "source_pack" / "raw"
PACK = ROOT / "source_pack"

# arXiv 正文里有大量换行与 LaTeX 残留，先做轻量清洗
_RE_HEADING = re.compile(r"^\s*\d+\s*$")
_RE_WS = re.compile(r"[ \t]+")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_text(raw: str) -> str:
    """把 arXiv 的换行版正文压成连续文本，保留段落感（换行 -> 空格）。"""
    text = raw.replace("\n", " ")
    text = _RE_WS.sub(" ", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    """切句。保留句末标点，便于逐字回查。"""
    text = text.strip()
    if not text:
        return []
    # 先按已有句末标点切，剩余长句再按分号等细分
    rough = re.split(r"(?<=[.!?])\s+", text)
    out: list[str] = []
    for seg in rough:
        seg = seg.strip()
        if not seg:
            continue
        if len(seg) > 600:
            # 超长句按逗号切分，避免单条 citation 无法定位
            parts = re.split(r"(?<=[,;:])\s+", seg)
            buf = ""
            for p in parts:
                if len(buf) + len(p) + 1 > 400 and buf:
                    out.append(buf.strip())
                    buf = p
                else:
                    buf = f"{buf} {p}".strip()
            if buf:
                out.append(buf.strip())
        else:
            out.append(seg)
    return out


def main() -> int:
    files = sorted(RAW.glob("arxiv_*.parquet"))
    if not files:
        print(f"ERROR: {RAW} 下没有 arxiv_*.parquet，请先下载。")
        return 1

    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"原始文档数：{len(df)}")

    docs = []
    for _, r in df.iterrows():
        entry_id = str(r["entry_id"])
        # http://arxiv.org/abs/2301.11659v2 -> 2301.11659v2
        doc_id = entry_id.rsplit("/abs/", 1)[-1]
        title = str(r["title"]).strip()
        text = normalize_text(str(r["text"]))
        sents = split_sentences(text)
        if len(sents) < 5:
            continue
        authors = r["authors"]
        if not isinstance(authors, (list, tuple)):
            authors = [str(authors)]
        cats = r["categories"]
        if not isinstance(cats, (list, tuple)):
            cats = [str(cats)]
        docs.append({
            "doc_id": doc_id,
            "title": title,
            "authors": [str(a) for a in authors],
            "primary_category": str(r["primary_category"]),
            "categories": [str(c) for c in cats],
            "published": str(r["published"]),
            "url": f"https://arxiv.org/abs/{doc_id}",
            "n_sentences": len(sents),
            "sentences": sents,
        })

    print(f"可用文档数（正文句子 >= 5）：{len(docs)}")

    out_docs = PACK / "documents.jsonl"
    with out_docs.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"已写出：{out_docs}")

    raw_meta = []
    for p in files:
        raw_meta.append({
            "file": p.name,
            "sha256": sha256_of(p),
            "bytes": p.stat().st_size,
        })

    cats = Counter(d["primary_category"] for d in docs)
    index = {
        "source_pack_id": "at3-03-arxiv-2023q1",
        "built_by": "build_source_pack.py",
        "dataset": {
            "name": "RealTimeData/arxiv_alltime",
            "repo_id": "RealTimeData/arxiv_alltime",
            "source_url": "https://huggingface.co/datasets/RealTimeData/arxiv_alltime",
            "mirror_url_used": "https://hf-mirror.com/datasets/RealTimeData/arxiv_alltime",
            "subsets_used": ["2023-01", "2023-02", "2023-03"],
            "license": "cc-by-nc-sa-4.0",
            "license_note": "arXiv 论文元数据与正文；原数据集声明为 CC BY-NC-SA 4.0",
            "downloaded_at": "2026-11-24T18:15:00+08:00",
            "raw_files": raw_meta,
        },
        "n_documents": len(docs),
        "n_categories": len(cats),
        "top_categories": [{"category": c, "count": n} for c, n in cats.most_common(20)],
        "doc_id_scheme": "arXiv entry_id 去掉 'http://arxiv.org/abs/' 前缀，例 2301.11659v2",
        "sentence_scheme": "正文按句末标点切分，句内保留原文；citation 以 (doc_id, s_idx) 定位",
        "documents_file": "documents.jsonl",
        "documents_sha256": sha256_of(out_docs),
        "offline": True,
        "network_used_at_runtime": False,
    }
    out_index = PACK / "index.json"
    out_index.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写出：{out_index}")
    print(f"documents.jsonl SHA256 = {index['documents_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
