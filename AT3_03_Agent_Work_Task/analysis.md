# AT3 ③ 分析報告（第二輪，最終提交版）

> 本檔為教師工作紙要求之「兩頁以內分析報告」，涵蓋：
> 完成狀態、tool calls 與 observations、最容易出錯的步驟、
> max_steps 小比較、第一輪 citation 失敗的根因與第二輪修正、仍然存在的限制。

---

## 1. 任務完成狀態

**完成。** `max_steps=6` 組的 final-state validator **8/8 全部通過**，
`max_steps=3` 組為 **3/8（FAIL）**。兩組均在 `results/` 下有獨立輸出目錄。

| 項目 | max_steps=3 | max_steps=6 |
|---|---|---|
| validator 總體 | **FAIL 3/8** | **PASS 8/8** |
| 工具呼叫 / 步數 | 3 / 3 | **5 / 6**（第 5 步即完成） |
| 讀取互異來源 | 2 | **3** |
| 簡報 | 未生成 | **已生成**（772 字元，3 條引用） |
| 終止原因 | `max_steps_reached` | **`brief_written`** |
| `lexical_overlap_rate` | —（無引用可算） | 1.0（診斷值） |

最終簡報的**三條引用全部通過逐字回溯**（`cited_sources/cited_sentences.json`
的 `all_verified: true`）：每個 marker 的 `(doc_id, sN)` 都能在來源包中定位到
該句原文，且該原文**逐字出現於帶該 marker 的那一條 bullet 內**（僅容許空白歸一化）。

---

## 2. 三篇來源及其選擇理由

任務主題為 **LLM Evaluation Methods**（評估對象與評估可靠性兩個維度）。
模型從 `search_sources(query="large language model evaluation", top_k=5)` 的
實際候選中，自主選出了三篇：

| doc_id | 論文 | 在簡報中的角色 | 引用句 |
|---|---|---|---|
| `2301.12004v1` | Understanding the Effectiveness of Very Large Language Models on Dialog Evaluation | **對話評估**：LLM 在對話任務上的評測有效性 | `#s4` |
| `2303.15078v3` | Large Language Models are Diverse Role-Players for Summarization Evaluation | **摘要評估**：以多角色提示機制做生成文本品質評估 | `#s5` |
| `2301.11916v3` | Large Language Models Are Latent Variable Models: Explaining and Finding Good Demonstrations for In-Context Learning | **輔助方法**：few-shot 示例選擇與 in-context 學習的敏感性（即評估用的 prompt 設計面向） | `#s1` |

**選擇理由（對應任務單 C2 的逐條要求）**：

- `2301.12004v1` 依指示作為**對話評估**來源，保留。
- `2303.15078v3` 依指示**必須讀取並檢查**，確認其確為直接相關的摘要評估候選
  （其 `s[5]` 即「既有指標與人工評測之間存在明顯落差」，正是評估可靠性的核心問題），
  故納入。
- `2301.11916v3` **僅在被明確定位為 few-shot demonstration selection /
  evaluation prompt 設計的輔助方法時**才可納入。本輪它被選中時所引用的是
  「此能力對 few-shot 示例的選擇高度敏感」——即**評估面向的 prompt 設計**，
  符合該條件，故納入。它不作為評估方法本身，而是評估方法在實作時所依賴的
  示例選擇機制。
- 另有一個候選 `2303.15621v2`（ChatGPT as a Factual Inconsistency Evaluator）
  屬更直接的第三篇評價論文，但**模型本輪未選它**；本實驗不替模型改選，
  如實記錄模型自主的選擇結果。
- **已禁止使用**第一輪的 `2301.11659v2`（Linear Algebra / Tensor Code to
  Hardware Accelerators），它與 LLM Evaluation 無直接關係。本輪該 id 因不在
  檢索候選內，被工具層 allowlist 直接拒絕。

---

## 3. 每個主要論斷的原文證據

最終簡報共三條 bullet，每條的原文證據如下（**逐字命中，經獨立複核**）：

| 簡報 bullet 的陳述 | 引用 | 來源原文（節錄） | 逐字 |
|---|---|---|---|
| 該 bullet 引用的第一篇句子 | `[doc_id:2301.12004v1#s4]` | "They achieve a high level of performance on various natural language processing (NLP) tasks such as question answering and summarization." | ✅ |
| 「既有指標與人工評測之間存在明顯落差」（評估可靠性） | `[doc_id:2303.15078v3#s5]` | "A big challenge to language evaluation is that there is a clear divergence between existing metrics and human evaluation." | ✅ |
| 「此能力對示例選擇敏感」（評估面向的 prompt 設計） | `[doc_id:2301.11916v3#s1]` | "However, existing literature has highlighted the sensitivity of this capability to the selection of few-shot demonstrations." | ✅ |

此外，簡報開頭的摘要句（"Large language models (LLMs) are increasingly being
evaluated across various tasks…"）與標題行屬模型的**概述性文字**，
依課程要求不強制引用；validator 的 `brief_exists` 只檢查其存在、標題與引用標記。

**重要界限**：逐字核驗證明的是「引文確為來源原句且確在該 bullet 內」，
**不等於**「該句在語義上充分支持該 bullet 的論斷」——後者超出本作業的自動驗證範圍。

---

## 4. Agent 的工具調用與 observation

### 4.1 max_steps=6 的實際調用序列（`results/run_maxsteps6/trajectory.jsonl`）

| step | 工具 | arguments | ok | observation 摘要 |
|---:|---|---|---|---|
| 1 | `search_sources` | `query="large language model evaluation", top_k=5` | ✅ | 5 個候選：doc_id + 標題 + 分類 + 3 條命中句片段；**不含全文** |
| 2 | `read_source` | `doc_id="2301.12004v1"` | ✅ | 全文句子清單（帶全局句號 `s[N]`），逐字可引用 |
| 3 | `read_source` | `doc_id="2303.15078v3"` | ✅ | 同上 |
| 4 | `read_source` | `doc_id="2301.11916v3"` | ✅ | 同上 |
| 5 | `write_brief` | `brief_text=...` | ✅ | "OK. brief written (772 chars), cited 3 distinct sources, 3 citation markers, all verified verbatim." |

**每一步的工具選擇、參數與引用句都由模型自主決定**：模型自行提出 query、
自行從候選中挑出三個 doc_id（每次挑不同的）、自行決定簡報文字與引用。
工具層只做三件事——allowlist（只能讀檢索回傳過的 id）、重複讀取保護、
以及寫作時的逐字核驗；**不代替模型選句、不自動補齊來源、不改寫 marker**。

`trajectory.jsonl` 與 `tool-results.jsonl` 的 `(step, tool, arguments)` 序列
已獨立交叉核對一致；`trajectory_consistent` 檢查亦驗證了
**最終寫入的簡報內容與日誌記錄的寫入內容一致**。

### 4.2 max_steps=3 的實際調用序列

| step | 工具 | arguments | ok |
|---:|---|---|---|
| 1 | `search_sources` | 同上 | ✅ |
| 2 | `read_source` | `2301.12004v1` | ✅ |
| 3 | `read_source` | `2303.15078v3` | ✅ |

步數耗盡，停在第二篇讀取後。**行為序列本身正確**——只是走不到寫作步。

---

## 5. 最容易出錯的步驟

**`write_brief` 是唯一會產出可被審查之文字的步驟，也是最易出錯的一步。**
其餘步驟（search / read）的失敗多為**參數層**問題，可被工具層直接攔截。

實測觀察到的失敗形態與對應的錯誤層級、恢復方式，完整記錄於 `error-note.md` §2.2。
其中最關鍵的四項：

| 失敗形態 | 錯誤層級 | 恢復方式 |
|---|---|---|
| 幻覺出不在檢索結果中的 doc_id | argument | 工具層 allowlist 拒絕 + 回灌候選清單 |
| 把論文**標題**當成 bullet 文字 | observation | 呈現層不提供標題作候選；Prompt 給出可判別啟發式 |
| 把來源句**改寫**成自己的概括 | observation | Prompt 要求逐字複製 + 逐字核驗拒絕 |
| 句子取自 A 篇、marker 卻掛到 B 篇 | observation / state update | 候選行把 doc_id 置於行首；核驗失敗時精確指出正確出處 |

**最脆弱的一環是「引用與論斷的語義對應」**：本輪門禁只做到逐字，
語義支持仍需人工審查（見 §7 限制 4）。

---

## 6. max_steps 小比較

**唯一變數是 `max_steps`**，其餘完全一致（已由兩組 `provenance.json` 逐欄核對）：

| 固定項 | 兩組相同值 |
|---|---|
| 任務目標 | `RESEARCH_TASK`（同一字串，來自 `contract.py` 單一來源） |
| 來源包 | `at3-03-arxiv-2023q1`，`documents.jsonl` SHA256 = `839583c4…fc4cd` |
| 模型 | Qwen2.5-Coder-3B-Instruct（bfloat16，greedy / temperature=0） |
| Prompt | `SYSTEM_PROMPT`（同一常數） |
| Tool Schema | `tool-schema.json`（同一來源，兩組由 `contract.py` 產生） |
| Validator | `validate_final_state.py` 同一份八項 gate |
| Evaluation rubric | `evaluation.csv` 同一份逐項評分 |
| 隨機種子 / 其他旗標 | seed=42，`api_used=false`、`network_used=false`、`framework` 相同 |

### 逐 gate 對照（`comparison_gates.csv`）

| gate | max_steps=3 | max_steps=6 | 隨 max_steps 改變 |
|---|---|---|---|
| brief_exists | FAIL | PASS | **是** |
| three_distinct_citations | FAIL | PASS | **是** |
| three_sources_read | FAIL | PASS | **是** |
| citations_traceable | FAIL | PASS | **是** |
| quoted_text_exists | FAIL | PASS | **是** |
| trajectory_consistent | PASS | PASS | 否 |
| within_step_budget | PASS | PASS | 否 |
| provenance_complete | PASS | PASS | 否 |

### 解讀（及其邊界）

五項「**工作完成度**」gate 隨 max_steps 改變；三項「**流程自律**」gate
（軌跡一致、步數合規、provenance 完整）兩組都成立。

這說明：**步數預算不影響 Agent 的行為自律性，但決定它能否把事做完**。
3 步組的失敗是**預算耗盡型**——它的三步行為序列與 6 步組的前三步**完全相同**
（同 query、同兩篇讀取順序、同參數），只是被預算截斷在寫作之前。
換言之，這不是「6 步讓 Agent 更聰明」，而是「5 步是完成這條最短工具鏈
（檢索 1 + 讀取 3 + 寫作 1）所需的最小預算」。

**明確聲明**：本比較**不得**被推廣為「Agent 智能水平隨步數提升」。
兩組的行為序列在前三步完全一致，差異純粹來自預算是否容得下寫作步。
6 步組在第 5 步即完成，第 6 步未使用。

---

## 7. 第一輪 citation 失敗的根因與第二輪的修正

第一輪結論標記為：**「程式形式驗證通過，但經人工審查發現 citation 語義支持不足，
不能作為最終提交版本。」**（完整證據保留於 `results/iteration_1/`）

### 7.1 根因（第一輪的四個實質缺陷）

1. **引用由工具端自動對齊生成**：`_autofill_sentence_ids` 以詞彙重疊在來源包中
   「找最佳來源句」並補上／替換 `#sN`。模型並未真正把某句原文抄進 bullet。
2. **自動補足第三篇來源**：模型只寫兩篇時，工具自動替它補上第三篇的 marker。
3. **門禁事後被放寬**：「引用句與簡報陳述的語義重疊率」最初為硬門禁，
   後降級為診斷指標（`semantic_alignment_rate`），而該指標本身只是詞彙重疊。
4. **來源選擇錯誤**：`2301.11659v2`（Linear Algebra / Tensor Code to Hardware
   Accelerators）與任務主題 LLM Evaluation 無直接關係。

### 7.2 第二輪的逐項修正（對應任務單 C3 / C4）

| 修正項 | 實現 | 可驗證證據 |
|---|---|---|
| 刪除 Prompt 中的真實 doc_id | 改為 `<doc_id_from_search_results>` 等佔位符，並說明參數必須來自最近一次檢索結果 | `grep -c "2301.11659v2" src/agent.py` = 0 |
| 候選 allowlist | `read_source` 僅接受曾由 `search_sources` 回傳的 doc_id | `src/test_citation_rules.py` T1.2 |
| 禁止自動對齊／補齊／改寫 | `_autofill_sentence_ids`、`_best_sentence` 整個方法刪除 | 同檔案中無 `autofill` 路徑 |
| 引用逐字核驗 | validator 要求來源句原文逐字出現於帶 marker 的那一行（僅容許空白歸一化） | validator 第 5 項；`quoted_text_exists` |
| 拒絕即不得自動改寫 | 核驗失敗回灌精確診斷（指出正確出處），由模型自行修正 | `error-note.md` §2.2 E7 |
| `semantic_alignment_rate` 改名 | 改為 `lexical_overlap_rate`，且**僅作診斷**，並在欄位說明中聲明「詞彙重疊不等於語義蘊含」 | `validator_maxsteps6.json` → `diagnostics` |
| `trajectory_consistent` 強化 | 比對 step + tool + **arguments**，並驗證**簡報寫入內容與日誌一致** | validator 第 6 項 detail |
| Prompt 與工具無真實 doc_id 洩漏 | Prompt 示例全部改為佔位符 | `grep` 結果為 0 |
| 保留真實 Agent 循環 | 未改為預設固定順序；模型仍自主決定 query、讀哪些、寫什麼 | `results/run_maxsteps6/trajectory.jsonl` |
| 引用規則單元測試 | 新增 `src/test_citation_rules.py`（16 項，無需 GPU） | 16/16 PASS |

### 7.3 仍需誠實說明的一點

第二輪達成 8/8 PASS 的關鍵，是工具層把候選**收斂到單句**並提供**可照抄的
bullet 模板**。這是工程輔助，**不是**模型能力的自然湧現——實測中 3B 模型在
候選較多時會同時犯「改寫」「錯配 doc_id」「三條取自同一篇」三類錯誤。
因此本結果**不可推廣**為「3B 模型具備可靠的引用生成能力」。

---

## 8. 仍然存在的限制

1. **單次運行，無跨次方差估計**。兩組各跑一次；`max_steps=6` 的成功不能
   推廣為必然。溫度 0 的貪心解碼使 run-to-run 方差應極小，但未實測。
2. **3B 模型的引用能力是主要瓶頸**（見 §7.3）。本輪的通過依賴工具層輔助。
3. **語義蕴含未驗證**。門禁只證「逐字」，不證「語義充分支持」。
   `lexical_overlap_rate` 是診斷值，**不具語義驗證效力**。
4. **來源包規模有限**：1,491 篇、127 個 primary_category；
   `cs.LG` 140、`cs.CV` 136、`quant-ph` 80、`cs.CL` 49
   （統計以 `source_pack/index.json` 為準）。主題簇密度限制了
   「讀三篇真正相關的評價論文」的可得性。
5. **GPU 熱降頻**：RTX 3070 Laptop 連續負載時時鐘 2100→約 1100–1500MHz，
   單次推理從約 4s 波動到 240s 以上；影響耗時，不影響結果內容。
6. **頁眉雜訊句**：arXiv 前幾句常含作者郵箱、日期戳；本輪在**呈現層**過濾，
   但**保留全局句號**，故 citation 座標與來源包完全一致。
7. **單一來源包**：未在其他主題簇上重複，步數效應是否跨任務穩定尚未驗證。

---

## 9. 下一步測試建議（每項只改一個變因）

1. **換模型**（其餘全不變）：Qwen2.5-Coder-7B-Instruct ——
   直接檢驗「引用品質的瓶頸是否在模型能力而非工具輔助」。
2. **移除可照抄模板**（其餘全不變）：量化工具層輔助對 PASS 的貢獻，
   讓「3B 模型的引用能力」成為可測量的量。
3. **擴到 500–1000 題的來源包**（其餘全不變）：提高主題簇密度，
   檢驗「三篇真正相關的評價論文」的可得性是否為限制 4 的成因。
4. **每組重複 3 次**（其餘全不變）：估計 run-to-run 方差，
   使限制 1 可被量化。
5. **補 `max_steps=4/5` 組**（其餘全不變）：找出「恰好能完成」的最小預算，
   檢驗本輪「5 步即足夠」的結論是否穩健。
6. **引入語義蕴含判定**（其餘全不變）：以 NLI 模型或人工審查補足
   限制 3，使「引用支持論斷」這一層也可被自動驗證。
