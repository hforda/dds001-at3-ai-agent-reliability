# AT3 ③ 用 Agent 完成一項工作（Offline Agent Lab）

> **課程**：DDS001 數據科學前沿分析研究｜**任課教師**：劉文堅
> **作業**：AT3 累積式大作業 Part ③（10%）
> **任務**：使用 Agent 搜尋指定來源包、讀取三個相關來源、生成一頁含 citation 的研究簡報，並通過 final-state validator。
> **受控比較**：`max_steps=3` vs `max_steps=6`（唯一變數）。

---

## 1. 實驗結果（真實運行）

| 組 | validator 總體 | 通過 | 工具呼叫 | 讀取來源 | 簡報 | 終止原因 |
|---|---|---|---:|---:|---|---|
| `max_steps=3` | **FAIL** | 3 / 8 | 3 | 2 | 否 | `max_steps_reached` |
| `max_steps=6` | **PASS** | **8 / 8** | 5 | 3 | 是 | `brief_written` |

`max_steps=6` 組的八項 gate 全部通過；三個 citation 皆可用 `(doc_id, s_idx)`
逐字回查至 `source_pack/documents.jsonl` 的原文句子。

**注意**：`max_steps=3` 的失敗是**預算耗盡型**（行為序列完全正確，只是走不到寫作步），
不是行為錯誤型 —— 這正是受控比較要觀察的效應。

---

## 2. 快速開始

```bash
# 0) 環境
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1) 指定本地模型權重（Qwen2.5-Coder-3B-Instruct，Apache-2.0）
#    權重不隨本專案發布，需自行從 Hugging Face 取得後指向該目錄。
export REACT_MODEL_PATH=/path/to/Qwen2.5-Coder-3B-Instruct

# 2) 零依賴自檢（來源包 / 模型 / 契約）
python src/run_lab.py --check

# 3) 單組實驗
python src/run_lab.py --run --max-steps 6 --out-dir results/run_maxsteps6

# 4) 受控比較（單一變數：max_steps）
python src/run_lab.py --compare --max-steps-a 3 --max-steps-b 6

# 5) 單獨重跑 validator
python src/validate_final_state.py --out-dir results/run_maxsteps6 --max-steps 6

# 6) 引用規則單元測試（無需模型與 GPU）
python src/test_citation_rules.py

# 7) 匯出被引用來源的完整原始記錄 + 逐字回溯校驗
python src/export_cited_sources.py
```

> **模型路徑解析順序**：`REACT_MODEL_PATH` → `REACT_MODEL_DIR` → `./models/Qwen2.5-Coder-3B-Instruct`
> → 同級工作區的 `AT3_02_技术复现/models/...`。前三者皆可獨立於本機路徑使用。
> 模型權重（數 GB）不隨本專案發布。

---

## 3. 來源包（學生自建，非教師 fixture）

| 項目 | 內容 |
|---|---|
| 資料集 | `RealTimeData/arxiv_alltime`（arXiv 論文元資料 + 正文） |
| 來源 URL | https://huggingface.co/datasets/RealTimeData/arxiv_alltime |
| 實際下載（鏡像） | `https://hf-mirror.com/datasets/RealTimeData/arxiv_alltime` |
| 使用分片 | `2023-01`、`2023-02`、`2023-03`（各約 12 MB，合計 35.9 MB） |
| 授權 | CC BY-NC-SA 4.0 |
| 文件欄位 | `entry_id`、`published`、`title`、`authors`、`primary_category`、`categories`、`text` |
| 建構結果 | **1,491 篇**可用文檔（正文句子 ≥ 5 者），每篇有唯一 `doc_id`（arXiv ID）與全文句子序列 |
| SHA256 | 見 `source_pack/index.json` 的 `raw_files[].sha256` 與 `documents_sha256` |

**doc_id 方案**：取 `entry_id` 去掉 `http://arxiv.org/abs/` 前綴（如 `2301.11659v2`），
可直接回溯到 `https://arxiv.org/abs/<doc_id>`。

**引用錨點**：正文按句末標點切分為帶全域句號 `s_idx` 的句子列表。
citation 格式 `[doc_id:2301.11659v2#s12]` 表示「該文件的第 12 句」，
validator 會回 `documents.jsonl` 逐字核對。

**離線保證**：實驗階段完全在本地讀 `source_pack/`，不發任何網路請求
（`provenance.json` 記錄 `network_used=false`，validator 會檢查此聲明與實測一致）。

---

## 4. Agent 設計

**真實工具調用循環** —— 每一步由本地模型根據「任務 + 工具說明 + 最近的 observation」
決定下一個動作，非預設的固定順序：

```
step 1  search_sources(query)      → 候選清單（僅索引層，不含全文）
step 2  read_source(doc_id)         → 全文句子（帶 s[] 編號）
step 3  read_source(doc_id)
step 4  read_source(doc_id)
step 5  write_brief(brief_text, cited_doc_ids) → 寫入 final-brief.md
```

三個工具的完整定義見 `tool-schema.json`（參數、回傳結構、副作用、是否計為一個 step）。

**模型**：Qwen2.5-Coder-3B-Instruct，本地權重，貪心解碼（temperature=0）。
無 API、無網路、無訓練、無微調、無大型 Agent framework。

### 4.1 工具層約束（第二輪新增）

這三項約束**不決定 Agent 的策略**，只劃定安全邊界：

| 約束 | 行為 | 目的 |
|---|---|---|
| **候選 allowlist** | `read_source` 僅接受曾由 `search_sources` 回傳的 doc_id | 防止模型憑記憶引用未檢索的文件 |
| **重複讀取保護** | 已讀過的 doc_id 再次請求即拒絕 | 避免步數預算浪費在無效動作 |
| **讀滿即停** | 讀滿 3 篇後 `read_source` 一律拒絕 | 把預算轉用於寫作；「讀哪三篇」仍由模型決定 |

### 4.2 Prompt 工程（第二輪）

- **佔位符取代真實 doc_id**：示例一律寫 `<doc_id_from_search_results>`，
  並說明參數必須來自最近一次檢索結果（`grep -c "2301.11659v2" src/agent.py` = 0）。
- **chat template**：用模型自帶的多輪對話結構，而非手工拼接純文字。
  實測手工拼接會誘發 3B 模型**模仿對話歷史**而無法產出新規劃。
- **逐字複製要求**：明示 bullet 必須是某條 `s[]` 行的逐字複製，
  並給出「像標題 = 錯／像論文句子 = 對」的判別啟發式。
- **頁眉雜訊過濾**：呈現層排除含郵箱／日期戳的 arXiv 頁眉句，
  但**保留其全局句號**，故 citation 座標與來源包完全一致。

**模型決策 vs 工具輔助的界線**（第二輪，已記錄）：
- **模型決定**：檢索什麼、讀哪三篇、簡報怎麼寫、用哪一句的內容與句號。
- **工具只做安全邊界與核驗**：候選 allowlist、重複讀取保護、讀滿即停、
  逐字核驗與精確診斷、括號配對解析、格式錯誤計數。

**工具端不做**（第一輪的缺陷，第二輪已全部移除）：
以詞彙重疊自動「找最佳來源句」、自動替換 `#sN`、自動補足第三篇來源。
核驗不過時**拒絕寫入**並回灌可操作診斷，由模型在下一步自行修正。
詳見 `error-note.md` §4。

**誠實聲明**：本輪能達成 8/8 PASS，關鍵在於工具層把候選收斂到單句並提供
可照抄的 bullet 模板。這是工程輔助，**不可推廣為 3B 模型具備可靠的引用生成能力**。

---

## 5. Validator（檢查最終交付狀態，非檔案是否存在）

`src/validate_final_state.py`，八項檢查：

| Gate | 判定依據 |
|---|---|
| `brief_exists` | 檔案存在、非空、≥400 字、有標題、含 citation 標記 |
| `three_distinct_citations` | 簡報中出現 **3 個互異** doc_id |
| `three_sources_read` | tool-results 中**成功**的 `read_source` 記錄，其 doc_id 互異數 ≥ 3 |
| `citations_traceable` | 每個 `(doc_id, s_idx)` 能在 `documents.jsonl` 中定位到該句 |
| `quoted_text_exists` | **逐字核驗**：該句原文須逐字出現在帶該 marker 的那一行 bullet 內（僅容許空白字符歸一化） |
| `trajectory_consistent` | trajectory 與 tool-results 的 `(step, tool, arguments)` 序列完全一致，且**最終寫入的簡報內容與日誌記錄一致** |
| `within_step_budget` | 工具呼叫數 ≤ `max_steps` |
| `provenance_complete` | 13 個必需欄位齊備，且 `api_used`/`network_used`/`max_steps` 與實測一致 |

另輸出**診斷指標**（不計入 PASS/FAIL）：`lexical_overlap_rate`，
量化「引用行與來源句的內容詞重疊率」。

> **名稱刻意為 lexical 而非 semantic**：詞彙重疊**不等於**語義蘊含驗證。
> 此指標僅作為模型事實性風險的量化參考；
> 引用有效性的判定依據是第 5 項的**逐字核驗**。
> （第一輪曾將此指標命名為 `semantic_alignment_rate` 並降級為診斷，
> 第二輪改名以免把詞彙重疊冒充語義驗證。）設計理由見 `error-note.md` §4。

---

## 6. 交付檔案

```
AT3_03_Agent_Work_Task/
├── final-brief.md              一頁研究簡報（3 個可回查 citation）
├── task-contract.json          任務契約：目標、約束、完成條件、validator gate 清單
├── tool-schema.json            三個工具的參數／回傳／副作用定義
├── trajectory.jsonl            逐步決策紀錄（thought / action / observation / 原始模型輸出）
├── tool-results.jsonl          每次工具呼叫的完整紀錄（含完整 observation）
├── evaluation.csv              匯總指標 + 八項 gate 逐項結果 + 逐步呼叫摘要
├── error-note.md               失敗定位、錯誤層級、recovery、安全停止、局限性
├── provenance.json             真實 provenance（fixture ID / 模型 / 網路 / SHA256 / 聲明）
├── analysis.md                 兩頁以內分析報告
├── comparison.csv              受控比較：兩組逐指標對照
├── comparison_gates.csv        受控比較：逐 gate 對照與是否隨 max_steps 改變
├── validator_maxsteps3.json    max_steps=3 組的 validator 輸出
├── validator_maxsteps6.json    max_steps=6 組的 validator 輸出
├── src/                        實作（8 個模組，見下）
├── source_pack/                離線來源包（documents.jsonl / index.json / raw/ / 授權說明）
├── cited_sources/              被引用三篇來源的完整原始記錄（見 §6.1）
├── history/                    歷史文件（第一輪證據、診斷執行、舊版計畫；不屬最終提交）
├── results/                    兩組實驗的完整輸出目錄
│   ├── iteration_1/            第一輪原始證據（完整保留，未刪除未篡改）
│   ├── run_maxsteps3/          max_steps=3 組輸出
│   └── run_maxsteps6/          max_steps=6 組輸出
├── AT3_03_source_pack_runtime.zip       運行所需來源包（documents.jsonl + index.json）
├── AT3_03_source_pack_raw_parquet.zip   原始 parquet 重建附件
└── requirements.txt            Python 依賴
```

### 6.0 history/（歷史文件，不屬最終提交）

| 內容 | 說明 |
|---|---|
| `iteration_1/` | 第一輪 `max_steps=3/6` 的完整原始證據。**第一輪結論：程式形式驗證通過，但經人工審查發現 citation 語義支持不足，不能作為最終提交版本。** 未刪除、未篡改、未重跑 |
| `cited_sources_iteration1/` | 第一輪引用的三篇（含與主題無關的 `2301.11659v2`），僅作對照 |
| `error-note_iteration1.md`、`analysis_iteration1.md` | 第一輪的分析文件 |
| `diagnostic_runs/` | 第二輪開發期間的 15 次診斷執行（完整保留失敗證據） |
| `AT3_03_执行计划.md` | 第一階段的執行計畫，**已被最終實驗取代**（當時假設教師 W11 student-kit 存在） |

### 6.1 cited_sources/（被引用來源的完整原始記錄）

由 `src/export_cited_sources.py` 從 `final-brief.md` 的 citation marker **自動抽取**（只讀導出，
不改動來源包任何檔案）。句子編號 `[sN]` 沿用 `source_pack/documents.jsonl` 的 `sentences[]`
全局下標，與 citation marker `[doc_id:XXXX#sN]` **嚴格同一座標**。

| 檔案 | 說明 |
|---|---|
| `2301.11659v2.md` | 276 句全文，引用句 `[s260]` 已標註 |
| `2301.11916v3.md` | 223 句全文，引用句 `[s9]` 已標註 |
| `2301.12004v1.md` | 235 句全文，引用句 `[s200]` 已標註 |
| `cited_sources.jsonl` | 三篇完整記錄（機器可讀，每行一篇） |
| `cited_sentences.json` | 被實際引用的句子清單 + 逐字回溯校驗結果 |
| `manifest.json` | 校驗和、來源資訊、句子編號約定 |

重跑：`python src/export_cited_sources.py`

**逐字回溯校驗**：腳本以與 `build_source_pack.py` 完全相同的預處理（`normalize_text`）
與切句規則，從原始 parquet 正文重新切句，核對 `sN` 處是否逐字一致。
三條引用**全部通過**（`all_verified: true`）。

### src/ 模組

| 檔案 | 職責 |
|---|---|
| `run_lab.py` | 主入口：`--check` / `--run` / `--compare`；載入模型、跑 Agent、寫出所有交付檔、執行 validator |
| `agent.py` | Agent 工具調用循環（chat template 多輪結構）；動作解析（括號配對、單雙引號、截斷、圍欄還原）；**嚴格逐字引用核驗**；錯誤回灌管理 |
| `tools.py` | 三個工具的實作與來源包檢索（BM25-lite）；`SourcePack` 記憶體索引；**候選 allowlist 與重複讀取保護** |
| `contract.py` | `task-contract.json` 與 `tool-schema.json` 的單一來源（保證兩組實驗完全一致） |
| `validate_final_state.py` | final-state validator（八項 gate + 診斷指標 `lexical_overlap_rate`） |
| `build_source_pack.py` | 由原始 parquet 建構來源包（切句、SHA256、索引） |
| `export_cited_sources.py` | 匯出被引用來源的完整原始記錄（保留句子編號）+ 逐字回溯校驗 |
| `test_citation_rules.py` | 引用規則與 allowlist 的單元測試（16 項，無需模型與 GPU） |

---

## 7. Provenance 與誠實聲明

- **來源包為學生自建**：`provenance.json` 的 `source_pack_built_by_student: true`
  與 `source_pack_disclaimer` 明確標註「非教師發放的 fixture」。
- **無付費 API、無網路、無訓練**：`api_used=false`、`network_used=false`、
  `gpu_used=true`（僅推理）、`agent_model_trained_by_us=false`。
- **結果不是模型效能基準**：本實驗記錄的是單一研究任務下的 Agent 行為軌跡，
  不可推廣為任何模型的通用能力陳述。
- **未使用**：AT3② 的 HotpotQA 資料、W09/W10 student-kit、博士研究與旅遊／口岸資料。

## 8. 授權

- 本工程為課程作業，**未**經授權不推送至任何公開倉庫。
- 模型權重 Qwen2.5-Coder-3B-Instruct 遵循 Apache-2.0。
- `RealTimeData/arxiv_alltime` 遵循 CC BY-NC-SA 4.0；arXiv 論文版權歸各自作者。
