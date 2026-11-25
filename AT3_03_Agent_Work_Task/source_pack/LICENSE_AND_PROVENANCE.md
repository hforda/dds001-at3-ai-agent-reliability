# AT3 ③ 離線來源包（Source Pack）說明與授權

## 一、這是什麼

本目錄是 AT3 ③「用 Agent 完成一項工作」的**離線來源包**：
Agent 在實驗中只能讀取本目錄的 `documents.jsonl`，**不得在執行期間連網**。

## 二、來源與授權

| 項目 | 內容 |
|---|---|
| 資料集名稱 | `RealTimeData/arxiv_alltime` |
| 來源 URL | https://huggingface.co/datasets/RealTimeData/arxiv_alltime |
| 實際下載來源（鏡像） | `https://hf-mirror.com/datasets/RealTimeData/arxiv_alltime` |
| 使用分片 | `2023-01`、`2023-02`、`2023-03`（arXiv 收錄年月） |
| 資料集授權 | **CC BY-NC-SA 4.0**（Creative Commons Attribution-NonCommercial-ShareAlike） |
| 本機建構者 | 學生（`src/build_source_pack.py`） |
| 建構結果 | **1,491 篇**可用文檔（正文句子 ≥ 5 者），127 個 primary_category |

### 授權條款摘要（CC BY-NC-SA 4.0）

- **BY（署名）**：使用或再散布本來源包時，須標示出處
  （`RealTimeData/arxiv_alltime`，arXiv 論文元資料與正文）。
- **NC（非商業）**：不得用於商業目的。本作業為課程作業，屬非商業使用。
- **SA（相同方式共享）**：若再散布，須以相同授權釋出。

### 重要聲明

1. **這不是教師發放的 fixture。** 課程任務書提及「核心 fixture」，但本機並無
   教師指定的 W11 來源包；本來源包由學生自行從 Hugging Face 公開資料集建構。
   此事實在 `provenance.json` 的 `source_pack_built_by_student: true` 與
   `source_pack_disclaimer` 中明確記錄，未隱瞞。
2. **每篇文檔的原始內容屬 arXiv 作者**，各自另有其授權。本包僅作為
   課程作業中的離線檢索對象。
3. `doc_id` 即 arXiv ID（如 `2301.12004v1`），可回溯至
   `https://arxiv.org/abs/<doc_id>` 查閱原文。

## 三、檔案內容

| 檔案 | 說明 |
|---|---|
| `documents.jsonl` | 每行一篇文檔，欄位：`doc_id`、`title`、`authors`、`primary_category`、`categories`、`published`、`url`、`n_sentences`、`sentences[]` |
| `index.json` | 來源包元資料：ID、建構方式、文件數、分類統計、`documents_sha256`、離線標記、授權 |

`sentences[]` 的下標即為 **citation 的全域句號**（`[doc_id:X#sN]` 中的 N），
從 0 開始計數，與 `export_cited_sources.py` 匯出的完整記錄完全一致。

## 四、兩個壓縮包

| 壓縮包 | 內容 | 大小 | 用途 |
|---|---|---|---|
| `AT3_03_source_pack_runtime.zip` | `documents.jsonl` + `index.json` | ~21 MB | **執行實驗所需**，可直接解壓後離線運行 |
| `AT3_03_source_pack_raw_parquet.zip` | 三個原始 parquet 分片 | ~32 MB | **重建附件**，供他人從原始資料重新建構並核對 |

### 重建方式

```bash
# 解壓原始分片後
python src/build_source_pack.py
# 會重建 source_pack/documents.jsonl 與 index.json，
# 並記錄各檔案的 SHA256 以供核對
```

## 五、校驗和（SHA256）

### 運行檔案

| 檔案 | SHA256 |
|---|---|
| `documents.jsonl` | `839583c4aeb23378dec3fbd7351844904fe513e9bc4bade56d3311f6085fc4cd` |
| `index.json` | `dea210f72189fff20a265d9562847aeac859270fb11b65328a479bb97eeb9cf5` |

### 原始分片（重建附件）

| 檔案 | SHA256 |
|---|---|
| `arxiv_2023-01.parquet` | `08463951e097d10adb828c0cd5823bbf29de1630675f8d7c6ee6d2c6e5ea8bee` |
| `arxiv_2023-02.parquet` | `c215c5f19b793da4599a5e8d6f20d33516ce76f5ce1771705a1cb09556cdae47` |
| `arxiv_2023-03.parquet` | `27197911c96822a1ccbbedfd23fb7a39c25a0fd9620b2bb161387e004c19c215` |

## 六、分類統計（以 `index.json` 為準）

前 20 個 primary_category：

| 排名 | 分類 | 篇數 |
|---:|---|---:|
| 1 | cs.LG | 140 |
| 2 | cs.CV | 136 |
| 3 | quant-ph | 80 |
| 4 | cs.CL | 49 |
| 5 | hep-ph | 37 |

共 127 个 primary_category，1,491 篇文檔。

## 七、發布限制

- 本來源包**可以**隨專案發布（CC BY-NC-SA 4.0 允許非商業再散布，須保留本授權說明）。
- 本目錄**不包含**：模型權重（Apache-2.0，數 GB）、AT3② 的 HotpotQA 原始資料、
  Wikipedia 快取、完整 arXiv 文獻庫。這些均不推送至公開倉庫。
