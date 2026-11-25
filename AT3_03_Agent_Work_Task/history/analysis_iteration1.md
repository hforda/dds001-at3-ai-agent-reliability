# AT3 ③ 實驗分析（analysis.md）

**任務**：用 Agent 完成一項離線來源整理工作——檢索指定來源包、讀取三個相關來源、生成一頁含 citation 的研究簡報，並通過 final-state validator。
**來源包**：`RealTimeData/arxiv_alltime`（2023-01～03 三個分片，1,491 篇 arXiv 論文，學生自建，非教師 fixture）
**模型**：Qwen2.5-Coder-3B-Instruct（本地權重，AT3 ② 已下載；無 API、無網路、無訓練）
**受控比較**：`max_steps=3` vs `max_steps=6`，其餘全部相同
**日期**：2026-11-24

---

## 一、完成狀態

| 組 | validator 總體 | 通過項 | 工具呼叫 | 讀取互異來源 | 簡報 | 終止原因 |
|---|---|---|---:|---:|---|---|
| `max_steps=3` | **FAIL** | 3 / 8 | 3 | 2 | 否 | `max_steps_reached` |
| `max_steps=6` | **PASS** | **8 / 8** | 5 | 3 | 是 | `brief_written` |

`max_steps=6` 組的八項檢查全部通過，逐項結果：

| Gate | 結果 | 具體內容 |
|---|---|---|
| `brief_exists` | PASS | 941 字元、有 `#` 標題、含 citation 標記 |
| `three_distinct_citations` | PASS | 3 個互異 doc_id：`2301.12004v1`、`2301.11916v3`、`2301.11659v2` |
| `three_sources_read` | PASS | 同上 3 個，且皆在 tool-results 中有成功的 `read_source` 觀測 |
| `citations_traceable` | PASS | 3 個 marker 的 `(doc_id, s_idx)` 全部能在來源包中定位到該句 |
| `quoted_text_exists` | PASS | 引文內容與簡報引用行有共同實質內容詞 |
| `trajectory_consistent` | PASS | trajectory 5 條與 tool-results 5 條的 (step, tool) 序列完全一致 |
| `within_step_budget` | PASS | 工具呼叫 5 次 ≤ 預算 6 |
| `provenance_complete` | PASS | 13 個必需欄位齊備；`api_used=false`、`network_used=false` 與實測一致 |

**任務完成**。`final-brief.md` 為一頁、主題為「LLM 評估方法」的研究簡報，
三個 bullet 分別引用三篇不同論文，每個引用都可用 `(doc_id, s_idx)` 逐字回查至
`source_pack/documents.jsonl` 的原文句子，並可再經 `https://arxiv.org/abs/<doc_id>`
回到原始論文頁面。

---

## 二、Tool calls 與 observations

`max_steps=6` 組的完整序列（見 `trajectory.jsonl` 與 `tool-results.jsonl`）：

| step | 工具 | 參數 | 結果 |
|---:|---|---|---|
| 1 | `search_sources` | `query="large language model evaluation methods", top_k=5` | 5 個候選（**僅索引層，不含全文**） |
| 2 | `read_source` | `doc_id=2301.12004v1` | 全文 24 句（`cs.CL`，Dialog Evaluation 與 LLM 有效性） |
| 3 | `read_source` | `doc_id=2301.11916v3` | 全文 24 句（`cs.CL`，LLM 作為潛變數模型／ICL 機制） |
| 4 | `read_source` | `doc_id=2301.11659v2` | 全文 24 句（`cs.LG`，線性代數與張量程式碼的硬體加速器匹配） |
| 5 | `write_brief` | `brief_text=...`, `cited_doc_ids=[3 個]` | 簡報寫入，3 個 marker |

**設計上刻意讓 `search_sources` 不回傳全文**，只有 `read_source` 才給正文。
這迫使 Agent 必須真的呼叫讀取工具，也讓「讀取三個來源」成為可觀測、可驗證的事實，
而不是模型的一句宣稱。`three_sources_read` 這項 gate 正是直接檢查 tool-results 中
**成功**的 `read_source` 記錄，而非讀模型的自述。

值得記錄的一點：檢索結果的前兩名（`2301.12004v1` 對話評估、`2301.11916v3` ICL 機制）
與第三名（`2301.11659v2` 硬體加速器）**主題並不完全一致**——第三篇是模型自行挑選的
（檢索命中的干擾項），這說明 3B 模型的來源取捨仍帶有隨機性。最終簡報仍把三篇
organize 成「評估對象」與「評估可靠性」兩條線，屬於模型自身的組織判斷。

---

## 三、最容易出錯的步驟

**`write_brief` 是出錯率最高的步驟**，且失敗高度集中。完整分析見 `error-note.md`，
此處摘要四類真實失敗與對應的錯誤層級：

| 失敗形態 | 錯誤層級 | 錯誤訊息（實測） | Recovery |
|---|---|---|---|
| 模型幻想多輪對話（`Assistant:`/`Human:`），多個 Action 混雜 | tool choice | 動作被誤判，浪費步數 | 取第一個可完整解析的 Action（零推理成本） |
| 單引號 / 輸出截斷導致閉引號遺失 | argument | `missing_doc_id`、`missing_brief_text` | 括號配對掃描 + 截斷補救（機械解析，不重生成） |
| 引用標記缺失或掛錯句子 | observation | `no_citation_markers`、`declared_but_not_cited` | 以 bullet 原文機械反查句號（模型仍決定用哪句） |
| 未讀滿三源就寫簡報 | state update | `fewer_than_3_citations` | 移除被拒內容的回灌，避免模型複製同一份無引用簡報 |

**最重要的一點**：這四類失敗全部可以在**不重新生成**的前提下 recovery。工具層把
「模型的決策」與「機械對齊」嚴格分開——模型決定檢索什麼、讀哪三篇、簡報怎麼寫；
工具只負責把「模型抄的原文」映射為「全域句號」。這個分界讓 3B 模型在受限條件下
仍能完成任務，同時每一個引用都保有可逐字回查的證據鏈。

---

## 四、受控比較結果

**唯一變數：`max_steps`（3 → 6）。** 任務、來源包、模型、prompt、tool schema、
validator、評價標準全部相同；兩組在**獨立的輸出目錄**執行，互不覆蓋。

| 指標 | max_steps=3 | max_steps=6 | 差異解讀 |
|---|---:|---:|---|
| validator 總體 | FAIL | **PASS** | **步數預算決定任務能否完成** |
| 通過 gate 數 | 3 / 8 | **8 / 8** | 5 個 gate 的結果隨步數改變 |
| 實際工具呼叫 | 3 | 5 | 6 步組在 5 步即完成，預算尚有餘裕 |
| 讀取互異來源 | 2 | **3** | 3 步只讀得完 2 篇 |
| 生成簡報 | 否 | **是** | 3 步組根本走不到寫作步 |
| 可回查 citation | 0 | **3** | 與簡報生成直接連動 |
| 終止原因 | `max_steps_reached` | `brief_written` | 失敗型態不同 |
| 耗時 | 25.7 s | 67.5 s | 步數越多，生成越多 |

逐 gate 的變化（`comparison_gates.csv`）：

| Gate | steps=3 | steps=6 | 是否隨 max_steps 改變 |
|---|---|---|---|
| `brief_exists` | FAIL | PASS | **是** |
| `three_distinct_citations` | FAIL | PASS | **是** |
| `three_sources_read` | FAIL | PASS | **是** |
| `citations_traceable` | FAIL | PASS | **是** |
| `quoted_text_exists` | FAIL | PASS | **是** |
| `trajectory_consistent` | PASS | PASS | 否（兩組都成立） |
| `within_step_budget` | PASS | PASS | 否（兩組都成立） |
| `provenance_complete` | PASS | PASS | 否（兩組都成立） |

**這個對照的意義**：三個「流程完整性」類 gate（trajectory 一致、步數合規、provenance 完整）
在兩組都成立，因為它們只檢查**執行是否自律**；而五個「工作完成度」類 gate 全部
隨步數改變。這說明步數預算的作用**不是**讓 Agent 做事更規矩，而是讓它**做得到
完整的事**。

`max_steps=3` 的失敗本質是**預算耗盡型**而非**行為錯誤型**：它的三步行為序列
（檢索 → 讀第 1 篇 → 讀第 2 篇）完全正確，只是任務鏈需要五步（1 檢索 + 3 讀取 + 1 寫作），
三步只能走到中途。這也是為什麼 `max_steps=3` 組的 `format_errors=0`、
`stop_reason=max_steps_reached` —— 它不是壞掉了，是沒錢走完。

---

## 五、實驗限制

1. **單一任務、單一模型、單次運行**。每組只跑一次，沒有跨多次運行的方差估計；
   貪心解碼（temperature=0）使 run-to-run 方差預期很小，但**未實測驗證**。
2. **3B 模型的能力上限**。實測顯示模型有三次真實失敗：寫出不存在的 doc_id
   （`2301.11660v1`）、引用標記不穩定輸出、bullet 用自己的概括語而非來源原句。
   前兩者已用 prompt 與解析層緩解；第三者的殘留影響以 `semantic_alignment_rate`
   診斷指標量化（本次 max_steps=6 組為 **1.0**，即三個引用的內容詞重疊全部達門檻）。
3. **機械對齊的介入**。`_autofill_sentence_ids` 會以 bullet 原文反查並補上／修正 `#sN`。
   這是索引層操作，**不產生新事實**，且引文仍須逐字存在於來源包（由 validator 驗證）；
   但它確實意味著「句號」不完全由模型獨立決定。此點已在 `task-contract.json` 與
   `error-note.md` §四明確記錄，不隱瞞。
4. **語意對齊率不設為門禁**。課程要求的是「引用內容是否真實存在於來源包」，
   與「引用句與簡報陳述的語意重疊率」並非同一件事。若以後者為硬門禁，
   3B 模型將無法產出簡報，實驗也將觀察不到步數效應。此為一項明確的設計決策，
   非遺漏。
5. **來源包規模有限**。1,491 篇涵蓋 2023 年第一季的 arXiv 論文，主題分佈以
   `cs.LG`(147)、`cs.CV`(136)、`quant-ph`(93) 等為主；`cs.CL` 僅 50 篇。
   任務主題選在「LLM 評估方法」是因為該主題在此範圍內仍有足夠密度的相關文檔。
6. **GPU 熱降頻**。RTX 3070 Laptop 在連續負載下時鐘由 2100 MHz 降至約 1100–1500 MHz，
   曾導致單次推理耗時從 4.4 s 暴增至 270 s。已加入 GPU 預熱（實測可把首次推理的
   一次性 kernel 編譯開銷排除在實驗耗時之外），但此為本機硬體限制，非實驗設計。
7. **結果不是模型效能基準**。本實驗記錄的是「單一研究任務下 Agent 的行為軌跡」，
   不可推廣為任何模型的通用能力陳述。

---

## 六、下一步測試建議

1. **只改模型規模**（首選）：改用 Qwen2.5-Coder-7B-Instruct（量化後約 5–6 GB，
   可容於本機 8 GB 顯存），其餘全部不變。預期觀察：`semantic_alignment_rate` 上升、
   `format_errors` 下降、步數使用量下降（不再需要試錯性呼叫）。
   這是唯一能直接檢驗「引用品質的瓶颈在模型能力而非實作」的實驗。
2. **只改任務**（次選）：換一個主題簇（例如 `cs.CV` 的檢索或分割評估），
   觀察步數效應是否在「檢索難度」不同的任務上保持同一方向。
3. **只改 validator 嚴格度**：把語意對齊率升為硬門禁，改用 7B 模型重跑，
   檢驗「更嚴的引用標準 + 更強模型」是否能同時滿足。
4. **只改步數**：補跑 `max_steps=4` 與 `max_steps=5`，找出「恰好能完成」的最小預算，
   這比 3 vs 6 的兩點比較更能刻畫預算-完成度的邊界。
5. **增加重複運行**：每組跑 3 次，估計 run-to-run 方差與隨機性對結論的影響。
