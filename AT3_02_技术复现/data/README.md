# 數據說明（AT3 ② ReAct 復現）

本目錄的數據全部為**公開可取**的第三方數據，本工程只做下載與抽樣，不做修改。
所有下載均經 `hf-mirror.com` 鏡像（本機 `huggingface.co` 不可達）。

---

## 1. HotpotQA distractor validation

| 項目 | 內容 |
|---|---|
| 數據集 | `hotpotqa/hotpot_qa`（HuggingFace 官方鏡像） |
| 配置 | **distractor**（每題含 10 個段落，含 2 個金標段落與 8 個干擾段落） |
| 劃分 | **validation** |
| 本地檔案 | `hotpot_qa_distractor_validation.parquet`（27.4 MB） |
| 題數 | **7,405** |
| 下載命令 | `curl -L https://hf-mirror.com/datasets/hotpotqa/hotpot_qa/resolve/main/distractor/validation-00000-of-00001.parquet` |
| 授權 | CC BY-SA 4.0 |

### 欄位

| 欄位 | 型別 | 說明 |
|---|---|---|
| `id` | str | 題目 ID |
| `question` | str | 問題文字 |
| `answer` | str | 金標答案 |
| `type` | str | `comparison` 或 `bridge`（多跳問題的兩種類型） |
| `level` | str | `easy` / `medium` / `hard` |
| `supporting_facts` | dict | 金標支持段落（`title` 與 `sent_id`），可用於診斷檢索是否命中 |
| `context` | dict | 10 個段落（2 金標 + 8 干擾）的標題與句子 |

### 抽樣方式（與原論文官方代碼一致）

官方 `hotpotqa.ipynb` 的抽樣：

```python
idxs = list(range(7405))
random.Random(233).shuffle(idxs)
idxs[:500]
```

本工程 `src/data.py::sample_subset` 採用**完全相同**的抽樣序列，僅取前 `n_questions` 題。
因此本次的 100 題**是官方 500 題子集的前綴**，可直接與官方結果做方向性對照。

### 為什麼用 validation 而非 train

- 原論文主表（Table 1）使用 HotpotQA **dev**（即 validation）集；
- few-shot 示例來自 **train** 集（6 題，人工撰寫軌跡），與評測集無重疊。

### 本工程未包含 train 集

few-shot 的 6 條示例**直接取自論文附錄 C.1 的原文**（含 question、thought、action、observation），
因此無需 train 集即可構造 prompt，且這比自行重新標註更忠於原文。

---

## 2. Wikipedia 內容

| 項目 | 內容 |
|---|---|
| 來源 | `https://api.wikimedia.org/core/v1/wikipedia/en`（Wikimedia 官方 REST API） |
| 為何不用 `en.wikipedia.org` | 本機網絡環境中主站全部超時（10s+），REST API 可達 |
| 授權 | CC BY-SA 4.0 |
| 快取目錄 | `data/cache/wikipedia/` |
| 快取理由 | ① 避免重複請求（每次 search/lookup 都要抓頁面）；② 保證同一次實驗內環境一致；③ 論文章附錄 A.2 已指出 Wikipedia 頁面隨時間變化會影響可復現性 |

### 快取檔案格式

每個檔案為一個 JSON，命名 `<kind>__<safe_key>.json`：

- `search__<entity>.json`：search API 的原始回應（`pages` 陣列，含 `title` / `excerpt`）
- `page__<title>.json`：page API 的原始回應（含 `source` wikitext）

### 環境的三種動作與其對應的 API

| ReAct 動作 | 實現 | 論文 §3.1 對應 |
|---|---|---|
| `search[entity]` | `GET /search/page?q={entity}&limit=6`；標題精確匹配則 `GET /page/{title}` 取正文前 5 句；否則回 top-5 相似實體 | 「returns the first 5 sentences … or else suggests top-5 similar entities」 |
| `lookup[string]` | 在 search 所得的同一頁正文中查找包含該字串的句子，回傳帶 `(Result i / n)` 前綴 | 「returns the next sentence in the page containing string, simulating Ctrl+F」 |
| `finish[answer]` | 直接結束，回傳 `Episode finished, reward = 1` | 「would finish the current task with answer」 |

### 與官方實現的差異（已在 `docs/paper_analysis.md` 聲明）

| 項目 | 官方 `wikienv.py` | 本工程 |
|---|---|---|
| search 命中判定 | 抓搜尋頁 HTML，比對 `div.mw-search-result-heading` 的 class | 用 REST API 的標題精確匹配（不區分大小寫） |
| search 正文 | 直接從搜尋結果摘要取 | 取頁面 wikitext，轉純文本後切句取前 5 句 |
| wikitext 處理 | 官方直接用 BeautifulSoup 抽文字 | 自寫 wikitext→純文本轉換（處理模板、引用、檔案連結、內部連結） |

---

## 3. 模型權重

| 項目 | 內容 |
|---|---|
| 名稱 | `Qwen/Qwen2.5-Coder-3B-Instruct` |
| 下載 | `https://hf-mirror.com/Qwen/Qwen2.5-Coder-3B-Instruct/resolve/main/<file>` |
| 本地路徑 | `models/Qwen2.5-Coder-3B-Instruct/` |
| 大小 | 約 6.2 GB（2 個 safetensors 分片） |
| 精度 | bfloat16 |
| 授權 | Apache-2.0 |
| 選型理由 | 3B 級可在 8GB 顯存以 bf16 全精度推理（無需量化，避免額外依賴）；Coder 系列在指令遵循與格式生成上表現較穩 |

### 為何不用 7B

7B 以 bf16 約需 15GB，超出 8GB 顯存；4-bit 量化需 bitsandbytes（Windows 支援不穩），
且量化會引入額外變數，使復現難以重現。故取 3B 全精度。

---

## 4. 資料完整性

```bash
# 校驗 HotpotQA
python -c "import pandas as pd; df=pd.read_parquet('data/hotpot_qa_distractor_validation.parquet'); print(df.shape)"

# 校驗模型完整性（應為 6,171,877,376 bytes）
python -c "
import json; d=json.load(open('models/Qwen2.5-Coder-3B-Instruct/model.safetensors.index.json'))
print(d['metadata']['total_size'])"
```
