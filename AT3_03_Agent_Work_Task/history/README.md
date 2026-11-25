# 歷史文件（已被最終實驗取代）

本目錄保存 AT3 ③ 早期階段的方案文件，**僅供審計與對照，不屬於最終提交內容**。

## 檔案

| 檔案 | 狀態 | 說明 |
|---|---|---|
| `AT3_03_执行计划.md` | **已被取代** | 第一階段的執行計畫。當時尚未確認來源包，且設計上假設「教師發放的 W11 student-kit」存在。 |

## 為何不屬於最終提交

1. 該計畫所依據的「教師指定來源包」在本機並不存在，後續改為由學生自建離線來源包
   （`RealTimeData/arxiv_alltime`，CC BY-NC-SA 4.0）。
2. 該計畫早於第一輪實驗的兩個關鍵修正：
   - 引用規則由「工具端自動對齊」改為「模型逐字複製 + 嚴格核驗」；
   - 工具層加入候選 allowlist 與重複讀取保護。
3. 其中的門禁設計（八項 gate）已在最終版本中落地，但引用核驗強度已顯著提高，
   該計畫書的描述已不再準確。

最終提交內容以專案根目錄的 `final-brief.md`、`task-contract.json`、`tool-schema.json`、
`trajectory.jsonl`、`tool-results.jsonl`、`evaluation.csv`、`error-note.md`、
`provenance.json`、`analysis.md` 以及 `results/` 為準。
