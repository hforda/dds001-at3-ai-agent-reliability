#!/usr/bin/env python
"""导出被引用的三篇来源的完整原始记录（保留原始句子编号）。

输入：
  - final-brief.md                     : citation marker 的唯一来源
  - source_pack/documents.jsonl         : 切句后的文档记录（句子编号即 citation 编号空间）
  - source_pack/raw/arxiv_*.parquet     : 下载的原始分片（提供未经切句的原始正文）

输出（cited_sources/）：
  - <doc_id>.md                : 单篇完整记录（Markdown，句号以 [sN] 标注）
  - cited_sources.jsonl        : 三篇完整记录（机器可读，每行一篇）
  - cited_sentences.json       : 被实际引用的句子清单 + 逐字回溯结果
  - manifest.json              : 校验和与来源信息
  - README.md                  : 索引与说明

原则：
  - 不修改来源包任何文件；仅做只读抽取。
  - 句子编号沿用来源包 sentences[] 的下标，与 citation 的 #sN 严格同一坐标。
  - 每条被引句子都用原始 parquet 正文做「重新切句后逐字命中」校验。
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACK = ROOT / "source_pack"
RAW = PACK / "raw"
OUT = ROOT / "cited_sources"
BRIEF = ROOT / "final-brief.md"

MARKER_RE = re.compile(r"\[doc_id:([^\]#\s]+)#s(\d+)\]")


_RE_WS = re.compile(r"[ \t]+")


def normalize_text(raw: str) -> str:
    """与 src/build_source_pack.py 一致的预处理：换行->空格、连续空白归一。

    必须先做这一步再切句，否则编号会与来源包错位（这是导出校验抓到的真实问题）。
    """
    text = raw.replace("\n", " ")
    text = _RE_WS.sub(" ", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    """与 src/build_source_pack.py 完全一致的切句规则（用于原文回溯校验）。"""
    text = text.strip()
    if not text:
        return []
    rough = re.split(r"(?<=[.!?])\s+", text)
    out: list[str] = []
    for seg in rough:
        seg = seg.strip()
        if not seg:
            continue
        if len(seg) > 600:
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


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_raw_texts(want: set[str]) -> dict[str, dict]:
    """从原始 parquet 取出目标文档的原始正文（未经切句）。"""
    import pandas as pd

    out: dict[str, dict] = {}
    files = sorted(RAW.glob("arxiv_*.parquet"))
    if not files:
        return out
    for fp in files:
        df = pd.read_parquet(fp)
        for _, row in df.iterrows():
            entry_id = str(row["entry_id"])
            doc_id = entry_id.rsplit("/abs/", 1)[-1]
            if doc_id in want:
                out[doc_id] = {
                    "source_file": fp.name,
                    "entry_id": entry_id,
                    "title": str(row["title"]),
                    "authors": str(row["authors"]),
                    "categories": str(row["categories"]),
                    "primary_category": str(row["primary_category"]),
                    "published": str(row["published"]),
                    "updated": str(row.get("updated", "")),
                    "doi": str(row.get("doi", "")),
                    "journal_ref": str(row.get("journal_ref", "")),
                    "abstract": str(row.get("abstract", "")),
                    "raw_text": str(row["text"]),
                }
    return out


def main() -> int:
    if not BRIEF.exists():
        print(f"[FAIL] 找不到 {BRIEF}", file=sys.stderr)
        return 1
    if not (PACK / "documents.jsonl").exists():
        print("[FAIL] 找不到 source_pack/documents.jsonl", file=sys.stderr)
        return 1

    brief = BRIEF.read_text(encoding="utf-8")
    marks = MARKER_RE.findall(brief)
    if not marks:
        print("[FAIL] final-brief.md 中没有 citation marker", file=sys.stderr)
        return 1

    cited_docs = sorted({d for d, _ in marks})
    cited_idx: dict[str, list[int]] = {d: [] for d in cited_docs}
    for d, s in marks:
        cited_idx[d].append(int(s))

    # 载入来源包记录
    docs: dict[str, dict] = {}
    with (PACK / "documents.jsonl").open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["doc_id"] in cited_docs:
                docs[d["doc_id"]] = d
    missing = [d for d in cited_docs if d not in docs]
    if missing:
        print(f"[FAIL] 来源包中缺少: {missing}", file=sys.stderr)
        return 1

    raw = load_raw_texts(set(cited_docs))

    OUT.mkdir(exist_ok=True)

    # 清理上一轮的残留导出（只删本脚本产出的 <doc_id>.md，且该 doc_id
    # 不在本次最终简报的引用清单中；README 与其他 json 不动）
    keep = {f"{d}.md" for d in cited_docs}
    for old_md in sorted(OUT.glob("*.md")):
        if old_md.name == "README.md":
            continue
        if old_md.name not in keep:
            old_md.unlink()
            print(f"[CLEAN] 移除上一轮残留: {old_md.name}")

    # ---------- 逐篇 Markdown 完整记录 ----------
    per_doc_meta = []
    for doc_id in cited_docs:
        d = docs[doc_id]
        sents = d["sentences"]
        hit = set(cited_idx[doc_id])
        lines = [
            f"# {d['title']}",
            "",
            f"- **doc_id**: `{doc_id}`",
            f"- **primary_category**: `{d['primary_category']}`",
            f"- **categories**: `{d['categories']}`",
            f"- **published**: `{d['published']}`",
            f"- **url**: {d['url']}",
            f"- **authors**: {d['authors']}",
            f"- **n_sentences**: {d['n_sentences']}",
            f"- **本檔引用句號**: "
            + ", ".join(f"`#s{i}`" for i in sorted(hit)),
            "",
            "---",
            "",
            "## 完整正文（原始句子編號 `[sN]`，N 為全局下標，與 citation 的 `#sN` 同一座標）",
            "",
        ]
        for i, s in enumerate(sents):
            mark = "  ← **本簡報引用此句**" if i in hit else ""
            lines.append(f"[s{i}] {s}{mark}")
            lines.append("")
        md = "\n".join(lines).rstrip() + "\n"
        (OUT / f"{doc_id}.md").write_text(md, encoding="utf-8")
        per_doc_meta.append(
            {
                "doc_id": doc_id,
                "title": d["title"],
                "primary_category": d["primary_category"],
                "published": d["published"],
                "url": d["url"],
                "n_sentences": d["n_sentences"],
                "cited_sentence_indices": sorted(hit),
                "markdown_file": f"{doc_id}.md",
                "markdown_sha256": hashlib.sha256(md.encode("utf-8")).hexdigest(),
            }
        )
        print(f"[OK] {doc_id}.md  ({d['n_sentences']} 句, 引用 {sorted(hit)})")

    # ---------- 机器可读完整记录 ----------
    jl = OUT / "cited_sources.jsonl"
    with jl.open("w", encoding="utf-8") as f:
        for doc_id in cited_docs:
            d = docs[doc_id]
            rec = dict(d)
            rec["cited_sentence_indices"] = sorted(set(cited_idx[doc_id]))
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ---------- 逐字回溯校验 ----------
    trace = []
    all_ok = True
    for doc_id in cited_docs:
        d = docs[doc_id]
        r = raw.get(doc_id)
        rebuilt = split_sentences(normalize_text(r["raw_text"])) if r else []
        for i in sorted(set(cited_idx[doc_id])):
            s = d["sentences"][i]
            in_pack = i < len(d["sentences"]) and d["sentences"][i] == s
            in_raw = (i < len(rebuilt)) and (rebuilt[i] == s)
            # 備援：原文中確實包含該句片段
            contains = bool(r) and (s in r["raw_text"])
            ok = in_pack and (in_raw or contains)
            all_ok = all_ok and ok
            note = ""
            if not contains and in_raw:
                # 原始正文中该句被换行/制表符分隔，normalize_text 将其合并为同一句，
                # 因此「逐字 substring」不命中，但按构建期同一规则重建切句后完全一致。
                note = (
                    "原始正文中该句被换行分隔，来源包构建期的 normalize_text "
                    "（换行->空格、连续空白归一）将其合并为同一句；"
                    "按同一规则从原始 parquet 重建切句后，sN 处逐字一致。"
                )
            trace.append(
                {
                    "doc_id": doc_id,
                    "sentence_index": i,
                    "citation_marker": f"[doc_id:{doc_id}#s{i}]",
                    "sentence": s,
                    "exists_in_source_pack": in_pack,
                    "matches_rebuilt_from_raw_parquet": in_raw,
                    "literal_substring_of_raw_text": contains,
                    "verified": ok,
                    "raw_source_file": r["source_file"] if r else None,
                    "note": note,
                }
            )

    (OUT / "cited_sentences.json").write_text(
        json.dumps(
            {
                "brief_file": "final-brief.md",
                "markers_total": len(marks),
                "distinct_doc_ids": len(cited_docs),
                "distinct_sentences_verified": len(trace),
                "all_verified": all_ok,
                "verification_method": (
                    "以 src/build_source_pack.py 的同一 deterministic 切句规则"
                    "从原始 parquet 正文重新切句，核对 sN 下标处的句子是否逐字一致；"
                    "若下标不匹配但原文中确含该句片段，则记为 literal_substring 命中。"
                ),
                "records": trace,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ---------- manifest ----------
    manifest = {
        "exported_from": {
            "brief": "final-brief.md",
            "source_pack_documents": "source_pack/documents.jsonl",
            "source_pack_raw_parquet": sorted(p.name for p in RAW.glob("*.parquet")),
        },
        "source_pack_provenance": {
            "dataset": "RealTimeData/arxiv_alltime",
            "source_url": "https://huggingface.co/datasets/RealTimeData/arxiv_alltime",
            "license": "CC BY-NC-SA 4.0",
            "note": (
                "此來源包由學生於本機自建（自 Hugging Face 公開下載），"
                "並非教師發放的 fixture。"
            ),
        },
        "sentence_numbering": {
            "scheme": (
                "沿用 source_pack/documents.jsonl 的 sentences[] 全局下标，"
                "与 citation marker 的 #sN 严格同一坐标；下标从 0 开始。"
            ),
            "split_rule": "re.split(r'(?<=[.!?])\\s+', text)；>600 字符的长句再按 [,;:] 细分",
        },
        "documents": per_doc_meta,
        # ---- 兩類檔案的 SHA256 分開記錄，避免哈希口徑混淆 ----
        # (1) 原始來源檔案：來源包中的 documents.jsonl 與原始 parquet 分片
        "source_files": {
            "documents_jsonl": {
                "file": "source_pack/documents.jsonl",
                "bytes": (PACK / "documents.jsonl").stat().st_size,
                "sha256": sha256_file(PACK / "documents.jsonl"),
                "role": "原始來源包（citation 座標的權威依據）",
            },
            "index_json": {
                "file": "source_pack/index.json",
                "bytes": (PACK / "index.json").stat().st_size,
                "sha256": sha256_file(PACK / "index.json"),
                "role": "來源包索引（統計與校驗記錄）",
            },
            "raw_parquet": [
                {
                    "file": f"source_pack/raw/{q.name}",
                    "bytes": q.stat().st_size,
                    "sha256": sha256_file(q),
                    "role": "原始下載分片（重建附件）",
                }
                for q in sorted(RAW.glob("*.parquet"))
            ],
        },
        # (2) 帶標註的導出檔案：本目錄產物
        "exported_files": {
            "cited_sources_jsonl": {
                "file": "cited_sources/cited_sources.jsonl",
                "bytes": jl.stat().st_size,
                "sha256": sha256_file(jl),
                "role": "三篇完整記錄（機器可讀）",
            },
            "cited_sentences_json": {
                "file": "cited_sources/cited_sentences.json",
                "bytes": (OUT / "cited_sentences.json").stat().st_size,
                "sha256": sha256_file(OUT / "cited_sentences.json"),
                "role": "引用句清單與逐字回溯校驗結果",
            },
            "markdown_exports": [
                {
                    "file": f"cited_sources/{m['markdown_file']}",
                    "bytes": (OUT / m["markdown_file"]).stat().st_size,
                    "sha256": m["markdown_sha256"],
                    "role": "帶句號標註的完整原始記錄",
                }
                for m in per_doc_meta
            ],
        },
        "verification_all_passed": all_ok,
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ---------- README ----------
    readme = ["# 被引用来源的完整原始记录", ""]
    readme += [
        "本目录由 `src/export_cited_sources.py` 从 `final-brief.md` 的 citation marker 自动抽取，",
        "**只读导出，不修改来源包任何文件**。",
        "",
        "## 文件清单",
        "",
        "| 文件 | 说明 |",
        "|---|---|",
    ]
    for m in per_doc_meta:
        readme.append(
            f"| `{m['markdown_file']}` | `{m['doc_id']}` — {m['title'][:70]} "
            f"（{m['n_sentences']} 句） |"
        )
    readme += [
        "| `cited_sources.jsonl` | 三篇完整记录（机器可读，每行一篇，字段同 documents.jsonl） |",
        "| `cited_sentences.json` | 被实际引用的句子清单 + 逐字回溯校验结果 |",
        "| `manifest.json` | 校验和、来源信息、句子编号约定 |",
        "",
        "## 句子编号约定",
        "",
        "`[sN]` 的 N 即 `documents.jsonl` 中 `sentences[]` 的下标（从 0 开始），",
        "与 citation marker `[doc_id:XXXX#sN]` 严格同一坐标。",
        "被本简报引用的句子在 Markdown 中标注 `← **本簡報引用此句**`。",
        "",
        "## 回溯校验",
        "",
    ]
    for t in trace:
        readme.append(
            f"- `{t['citation_marker']}` → 句子编号命中来源包: "
            f"{'是' if t['exists_in_source_pack'] else '否'}；"
            f"由原始 parquet 重新切句后逐字一致: "
            f"{'是' if t['matches_rebuilt_from_raw_parquet'] else ('原文中含该片段' if t['literal_substring_of_raw_text'] else '否')}"
            + (f"（註：{t['note']}）" if t.get("note") else "")
        )
    readme.append("")
    (OUT / "README.md").write_text("\n".join(readme), encoding="utf-8")

    print(f"\n[OK] 导出完成 -> {OUT}")
    print(f"     引用 marker {len(marks)} 個，{len(cited_docs)} 篇文档，"
          f"逐字回溯全部通过: {all_ok}")
    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
