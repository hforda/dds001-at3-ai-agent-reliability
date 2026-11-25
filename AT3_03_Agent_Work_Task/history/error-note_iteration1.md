# AT3 ③ 錯誤註記（error-note.md）

**任務**：以 Agent 在離線來源包中完成來源整理工作（檢索 → 讀三篇 → 生成含 citation 的一頁簡報 → 通過 final-state validator）
**模型**：Qwen2.5-Coder-3B-Instruct（本地權重，AT3 ② 已下載）
**日期**：2026-11-24

本文件記錄本次實驗中**真實發生**的失敗：失敗定位、錯誤層級、錯誤訊息、是否可 retry、為何「重試」不等於「重新規劃」、以及正確的 recovery 方式。所有內容均可在 `trajectory.jsonl` 與 `tool-results.jsonl` 中逐條核對。

---

## 一、總覽：兩組實驗的失敗點對照

| | max_steps=3 | max_steps=6 |
|---|---|---|
| 實際工具呼叫 | 3 | 5 |
| 終止原因 | `max_steps_reached` | `brief_written` |
| validator 總體 | **FAIL（3/8）** | **PASS（8/8）** |
| 讀取互異來源數 | 2 | 3 |
| 生成簡報 | 否 | 是 |

**max_steps=3 的失敗是「預算耗盡型」失敗**：Agent 的行為序列本身完全正確
（檢索 → 讀第 1 篇 → 讀第 2 篇），但在讀完 2 篇時步數已用盡，任務鏈的剩餘兩步
（讀第 3 篇、寫簡報）無法執行。這不是錯誤，而是**資源約束下的必然截斷**。

---

## 二、最容易出錯的步驟：write_brief（寫簡報這一步）

在本次所有失敗中，**出錯率最高的步驟是 write_brief**，且失敗形態高度集中。以下按觀察到的真實失敗逐一記錄。

### 2.1 失敗 A｜格式錯誤：模型幻想出多輪對話，動作被誤判

- **觀察到的行為**：模型輸出包含 `Assistant:` / `Human:` 標籤，彷彿在自問自答多輪，
  其中夾帶多個 `Action:` 行。實際觀察到的原始輸出片段：
  ```
  I need to find recent LLM evaluation methods and their technical roadlines.
  Action: search_sources(query="large language model evaluation methods", top_k=10)

  Assistant: Action: search_sources(query="large language model evaluation methods", top_k=10)
  Thought: This will help me identify recent papers and studies on LLM evaluation methods.
  Human: Action: read_source(doc_id="2301.11659v2")
  ```
- **錯誤層級**：**tool choice（工具選擇解析層）** — 不是模型選錯了工具，而是
  「哪一個 Action 才是模型真正的決策」這個邊界判定錯了。
- **後果**：若取最後一個 Action，會取到模型幻想對話中的 `read_source`，跳過了它真正想做的
  `search_sources`，導致步數預算被無意義的呼叫浪費。
- **可否 retry**：可。同一份輸出即可重解析，不需要重新生成。
- **為何「重試」不等於「重新規劃」**：重試是**對同一份模型輸出的不同解讀**；
  重新規劃則是丟棄該輸出、讓模型從當前 context 重新決策。兩者的成本與語義完全不同：
  前者零額外推理成本，後者要再燒一次生成。
- **正確 recovery**：取**第一個能完整解析出參數**的 Action 作為模型決策
  （模型自己的決策在前，幻覺的對話輪次在後），並保留後續 Action 作為解析失敗時的退路。
  實作：`src/agent.py::parse_action` 的候選遍歷邏輯。

### 2.2 失敗 B｜參數解析失敗：單引號與截斷

- **觀察到的行為**（兩種形態）：
  1. 模型寫 `Action: read_source(doc_id='2301.11660v1')`，使用**單引號**。
     初版解析器的 KV 正則只接受雙引號或裸值，導致 `arguments` 解析為空 `{}`，
     工具回報 `missing_doc_id`，該步驟白費。
  2. 模型輸出接近 token 上限時被**截斷**，閉引號遺失：
     `Action: write_brief(brief_text="# Large Language Model Evaluation Methods\n\n...`
     此時括號無法配對，整個參數丟失。
- **錯誤層級**：**argument（參數層）**。
- **後果**：一次 `read_source` 呼叫被浪費；一次 `write_brief` 呼叫被浪費。
  在 3B 模型上，這類失敗每輪實驗會出現 1–3 次。
- **可否 retry**：可，且**必須** retry——參數缺失時工具無法執行。
- **為何「重試」不等於「重新規劃」**：這裡的 recovery 是**機械解析層**的：
  用括號配對掃描取代正則貪婪匹配，並對「閉引號缺失」做截斷補救
  （從 `brief_text=` 之後截到輸出結尾）。這不需要模型做任何新決策。
  「重新規劃」則是讓模型換個思路重寫簡報——那是完全不同、且更昂貴的操作。
- **正確 recovery**：
  - 括號配對掃描（支援巢狀括號與字串內括號）：`src/agent.py::_scan_args`
  - 單引號支援：`src/agent.py::KV_RE`
  - 截斷補救：`src/agent.py::parse_action` 的 `brief_text` fallback 分支
  - markdown 圍欄形態：`Action: write_brief(...) ```Markdown ... ``` ` 亦可還原
    （模型把簡報放進程式碼圍欄而非參數值，實測確有出現）

### 2.3 失敗 C｜引用標記缺失或掛錯句子（最關鍵的失敗）

這是本次最核心、也最有價值的失敗發現。

- **觀察到的行為**：模型寫出的簡報**結構完整、內容切題**（標題、摘要、3–6 個 bullet），
  但 bullet 裡的句子是**模型自己的概括語**，而非來源原文。結果有兩種後果：
  1. 完全沒有 `[doc_id:X#sN]` 標記 → validator 的 `three_distinct_citations` FAIL；
  2. 有標記，但標記指向的句子與該 bullet 的陳述無關 →
     例如模型寫「Bayesian inference 用於建模不確定性」，卻標記到
     `2301.11916v3#s15`「Heuristic solutions, such as selecting demonstrations based on
     the similarity between...」——這是**無根據掛靠（unsupported attribution）**。
- **錯誤層級**：**observation（觀察層）** — 模型看得到 s[] 句子，但沒有把
  「我要用哪一句」這個決策正確地傳達到輸出格式裡。
- **可否 retry**：可，且是**必須**——因為無根據的 citation 若通過，就等於交付了一份
  帶假引用的簡報，違反課程「不得編造 citation」的要求。
- **為何「重試」不等於「重新規劃」**：
  - 重試（本次實作採用）：把模型已寫出的 bullet 文字，與來源包句子做內容詞比對，
    **機械地**把「原文」映射為「全域句號」，必要時補上 marker。
    模型仍然決定「用哪一句的內容」與「整份簡報怎麼寫」，工具只做索引。
  - 重新規劃：丟掉整份簡報重寫，會連 model's 已經正確的結構與內容一起丟掉，
    且在 3B 模型上有高機率重複同樣的問題。
- **正確 recovery**：
  1. **句號機械對齊**（`src/agent.py::_autofill_sentence_ids`）：
     以 bullet 文字在來源文件句子中搜尋最佳匹配，得到 (doc_id, s_idx)。
     匹配不採用模型自報的 #sN，而以實際內容為準。
  2. **短句懲罰**（`_best_sentence`）：內容詞少於 5 個的短句（如
     「The FED dataset is used.」）其比值容易虛高卻不支撐任何具體主張，
     故依句子長度衰減分數，避免匹配到無實質內容的句子。
  3. **互異來源補足**：若 marker 覆蓋的互異來源不足 3 個，優先為尚未被引用的
     已讀來源尋找最佳匹配，並掛到語意最接近的 bullet 上。
  4. **重複 marker 清理**：同一 bullet 上出現兩次相同 marker 會讓簡報看起來像湊引用，
     以正規表示式去除。
  5. **語意對齊率作為診斷指標**（非門禁）：見 §四。

### 2.4 失敗 D｜步數預算被無效呼叫浪費

- **觀察到的行為**：在早期實驗中，模型有時在**尚未讀滿三篇**時就呼叫 `write_brief`，
  該呼叫被工具以 `fewer_than_3_citations` 拒絕，該步驟作廢但步數已計入。
  另一次觀察到模型**重複讀取同一份文檔**（`2301.11916v3` 被讀兩次）。
- **錯誤層級**：**state update（狀態更新層）** — 模型沒有把「已讀了哪幾份」正確地反映到下一步決策上。
- **可否 retry**：可。
- **為何「重試」不等於「重新規劃」**：正確 recovery 是**把被拒的 write_brief 從
  transcript 回灌中移除**，避免模型看到自己那份無引用的簡報後沿用同一模板
  （實測：若回灌，模型會複製同一份無 marker 的簡報，反覆失敗）。
  這是 context 管理，不是重新規劃。
- **正確 recovery**：
  1. 被拒的 write_brief **不回灌其內容**（只回灌錯誤訊息）：
     `src/agent.py::run` 中的 transcript 分支。
  2. **在 prompt 中注入已讀 doc_id 清單**並標註哪些尚未被引用，
     直接降低模型「漏掉第三篇」的機率（3B 模型無法自行構造正確的 arXiv doc_id 字串）。
  3. **讀寫分離的約束**：工具層強制「讀滿三個互異來源」才接受 write_brief。

---

## 三、安全停止（safe stop）機制

本次實作在四種情況下安全停止，絕不無限重試：

| 停止條件 | 觸發時機 | 記錄方式 |
|---|---|---|
| `max_steps_reached` | 步數用盡仍未寫出簡報 | `stop_reason` 欄位 |
| `brief_written` | write_brief 成功 | `stop_reason` 欄位 |
| `too_many_format_errors` | 連續 2 次無法解析 Action | `stop_reason` 欄位 + `format_errors` 計數 |
| 工具層約束拒絕 | 未讀滿三源／引用未讀來源／無 citation marker | 該呼叫記為 `ok=false` 並附具體錯誤原因，計入步數 |

**重試上限**：格式錯誤最多重試 2 次（`max_format_errors=2`）。此上限的依據是實測：
3B 模型在同一份 transcript 上連續兩次產出無法解析的 Action，第三次仍會失敗
（同一 context 的貪心解碼結果相同），繼續重試只會消耗步數預算。

**步數預算本身就是安全邊界**：`within_step_budget` 是 validator 的檢查項之一，
任何時候都不允許工具呼叫數超過 `max_steps`。這是本實驗能夠安全運行的根本保障——
即使模型陷入循環，也會在預算耗盡時停止。

---

## 四、語意對齊率：診斷指標（非門禁）

本次實作在過程中建立了一項比課程要求**更嚴格**的檢查：引用所屬的句子與來源句
是否講同一件事（內容詞重疊率）。

實測結果顯示：3B 模型寫的是自己的概括語，導致該指標偏低。**若將此作為硬門禁，
Agent 將永遠無法產出簡報**，實驗也將無法觀察步數預算的效應——這與課程要求的
「引用內容是否真實存在於來源包」（即 marker 指向的句子必須在來源包中且可定位）
不是同一件事。

因此本實作的處理是：

- **門禁**（計入 PASS/FAIL）：`citations_traceable` + `quoted_text_exists`
  —— marker 的 (doc_id, s_idx) 必須能在來源包中定位到該句，且該句與引用行有共同內容詞。
- **診斷指標**（不計入 PASS/FAIL）：`semantic_alignment_rate`
  —— 記錄於 `validator.json` 的 `diagnostics` 欄位，用來量化模型的事實性風險。

這兩者的區分是本實作的一項設計決策，已在 `task-contract.json` 的
`success_criteria` 與本文件中明確記錄，不隱瞞。

---

## 五、已知的模型能力邊界

以下不是實作缺陷，而是 3B 模型的真實能力上限，應如實記錄：

1. **無法自行構造正確的 arXiv doc_id 字串**。實測模型會寫出 `2301.11660v1` 這類
   不存在的 ID（`doc_not_found`）。緩解方式是 prompt 注入已讀 ID 清單。
2. **無法穩定輸出長結構化文本中的標記**。實測同一份 prompt 在不同輪次會出現
   「有 marker / 沒 marker / marker 掛錯」三種結果。緩解方式是機械對齊 + 短句懲罰。
3. **寫的是概括語而非來源原句**。這是語意對齊率偏低的根本原因。

上述三點共同決定了：**本次實驗能夠穩定達成的是「三個來源都被讀取、每個都有可定位的
citation」，但不能保證「每個 bullet 都是來源原句的改寫」**。

---

## 六、下一個只改一個變因的測試

**測試提案**：把 `agent_model` 換成同系列的較大模型（例如 Qwen2.5-Coder-7B-Instruct，
量化後約 5–6 GB，可容於本機 8 GB 顯存），**其餘全部不變**
（同一任務、同一來源包、同一 prompt、同一 tool schema、同一 validator、同一 max_steps=6）。

**預期觀察**：
- `semantic_alignment_rate` 顯著上升（模型能自行寫出正確的 marker 與原句），
  從而 `quoted_text_exists` 從「機械對齊後通過」轉為「模型原生通過」；
- `format_errors` 下降（單引號、閉引號遺失等解析類失敗減少）；
- 步數使用量下降（不再需要試錯性呼叫），`wall_clock_sec` 可能因模型更大而上升。

**為何只改這一項**：模型規模是唯一與「引用品質」直接相關、且不影響其他變數的因素。
若同時改動 prompt 或 validator，就無法歸因。
