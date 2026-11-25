# 被引用来源的完整原始记录

本目录由 `src/export_cited_sources.py` 从 `final-brief.md` 的 citation marker 自动抽取，
**只读导出，不修改来源包任何文件**。

## 文件清单

| 文件 | 说明 |
|---|---|
| `2301.11659v2.md` | `2301.11659v2` — Matching Linear Algebra and Tensor Code to Specialized Hardware Accele （276 句） |
| `2301.11916v3.md` | `2301.11916v3` — Large Language Models Are Latent Variable Models: Explaining and Findi （223 句） |
| `2301.12004v1.md` | `2301.12004v1` — Understanding the Effectiveness of Very Large Language Models on Dialo （235 句） |
| `cited_sources.jsonl` | 三篇完整记录（机器可读，每行一篇，字段同 documents.jsonl） |
| `cited_sentences.json` | 被实际引用的句子清单 + 逐字回溯校验结果 |
| `manifest.json` | 校验和、来源信息、句子编号约定 |

## 句子编号约定

`[sN]` 的 N 即 `documents.jsonl` 中 `sentences[]` 的下标（从 0 开始），
与 citation marker `[doc_id:XXXX#sN]` 严格同一坐标。
被本简报引用的句子在 Markdown 中标注 `← **本簡報引用此句**`。

## 回溯校验

- `[doc_id:2301.11659v2#s260]` → 句子编号命中来源包: 是；由原始 parquet 重新切句后逐字一致: 是
- `[doc_id:2301.11916v3#s9]` → 句子编号命中来源包: 是；由原始 parquet 重新切句后逐字一致: 是
- `[doc_id:2301.12004v1#s200]` → 句子编号命中来源包: 是；由原始 parquet 重新切句后逐字一致: 是（註：原始正文中该句被换行分隔，来源包构建期的 normalize_text （换行->空格、连续空白归一）将其合并为同一句；按同一规则从原始 parquet 重建切句后，sN 处逐字一致。）
