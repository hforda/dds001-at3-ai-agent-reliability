# AT3 ② 技術復現：ReAct

> **論文**：ReAct: Synergizing Reasoning and Acting in Language Models
> Yao, Zhao, Yu, Du, Shafran, Narasimhan, Cao — **ICLR 2023**（In-Person Poster, top 5%）
> arXiv: https://arxiv.org/abs/2210.03629
>
> **本工程**：從零實現 ReAct 的核心方法與四條 prompting baseline，在 HotpotQA 上復現原論文 Table 1 的對照結構與 Table 2 的失敗模式分析。

---

## ⚠️ 誠實聲明（請先讀這一段）

本工程是**簡化復現**，不是完整復現，且**不得**被描述為「重現了原論文結果」。

| 項目 | 原論文 | 本工程 | 影響 |
|---|---|---|---|
| 模型 | PaLM-540B / GPT-3 text-davinci-002（閉源，付費） | Qwen2.5-Coder-3B-Instruct（開源，本地 8GB 顯存推理） | **絕對 EM 必然顯著偏低**，這是模型規模差異，非實現錯誤 |
| 官方代碼 | github.com/ysymyth/ReAct | **完全不可達**（本機 GitHub 網路被阻斷），故**全部從零實現** | 符合任務書「獨立完成核心方法的實現」要求 |
| Wikipedia 環境 | 直接抓 en.wikipedia.org | 改用 `api.wikimedia.org` 重建（主站不可達） | 語義等價，但檢索質量與 2023 年的頁面內容可能有差異 |
| 評測題數 | 500 題（官方 seed=233 抽樣） | 100 題（**同一抽樣序列的前綴**） | 100 題上 EM 的 95% CI 寬度約 ±9pp，小幅差異需配對檢驗才可解讀 |
| baseline 數 | 8 條（含 CoT-SC 與兩條組合） | 4 條（Standard / CoT / Act / ReAct） | CoT-SC 需 21 樣本採樣，成本與方差在 3B 上不可接受 |
| Table 2 標註 | **人工**標註 200 條 | **規則式自動**標註 | 口徑不同，僅可作**方向性**對照 |

**驗收標準**：本次復現的成功與否，**不**以「絕對 EM 是否接近 27.4 / 29.4 / 25.7 / 28.7」判定，而以
1. 四條 baseline 的**相對排序**是否與原文一致、
2. **Act vs ReAct** 這條最關鍵的受控對比的方向是否一致、
3. 失敗模式的**結構**是否重現（接地是否降低幻覺）判定。

---

## 1. 為什麼選這篇

在 AT3 ① 文獻綜述收錄的 8 篇核心 + 4 篇補充論文中，ReAct 是唯一同時滿足以下條件的候選
（完整評估見 `AT3_02_候选论文评估.md`）：

1. **核心方法純粹在算法層面** —— 動作空間擴充 $\hat{\mathcal{A}} = \mathcal{A} \cup \mathcal{L}$ 與上下文更新律
   $c_{t+1} = (c_t, \hat{a}_t)$，不依賴訓練、不依賴付費 API、不依賴外部授權；
2. **有可直接對照的定量結構** —— Table 1 的七條 baseline 實為同一批人工軌跡的系統性消融，
   四條件設計是它的最小子集；
3. **數據雙雙實測可得** —— HotpotQA distractor validation 已下載；`api.wikimedia.org` 實測可支撐
   `search` / `lookup` / `finish` 三動作；
4. **風險最低** —— 相較其他候選，其失敗模式僅為「模型能力不足」，可如實歸因於規模，而非讓實驗失去意義。

---

## 2. 快速開始

### 2.1 先決條件

- Python 3.13（本機 venv：`hk_adv311`）
- NVIDIA GPU（本次為 RTX 3070 Laptop，8GB）
- 網絡可達 `hf-mirror.com` 與 `api.wikimedia.org`；**GitHub 與 huggingface.co 在本機不可達**

### 2.2 安裝依賴

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> torch 需帶 CUDA 輪子（PyPI 上只有 CPU 版）。依本機 CUDA 版本選擇對應索引：
> ```bash
> pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
> ```
>
> **CUDA 版本口徑**：本實驗實際使用的是 **PyTorch 自帶的 CUDA Runtime**（`torch.version.cuda`
> = 12.6，對應 wheel `torch 2.14.1+cu126`），**不依賴系統 CUDA Toolkit**。
> 本機另有 NVIDIA 驅動 576.52（支援 CUDA 12.9）與系統 Toolkit 13.0（僅供編譯擴充，
> 本實驗未使用）。三者為不同概念，詳見 `environment.txt`。

### 2.3 下載數據與模型

```bash
bash scripts/download_data.sh
```

會下載：

- `data/hotpot_qa_distractor_validation.parquet`（27.4 MB，HotpotQA distractor validation，7,405 題）
- `models/Qwen2.5-Coder-3B-Instruct/`（約 6.2 GB）

> **注意**：本機 `huggingface.co` 不可達，腳本已默認使用 `hf-mirror.com` 鏡像。
> 若你的網絡可直連 huggingface.co，可設 `export HF_ENDPOINT=https://huggingface.co` 覆蓋。

### 2.4 運行

```bash
# 冒煙測試（2 題，只跑 ReAct）
python run_experiment.py --methods react --n-questions 2 --tag smoke

# 完整實驗（100 題 × 4 baseline）
python run_experiment.py --methods standard cot act react --n-questions 100 --tag main

# 生成對照表與圖表
python evaluate.py --tag main

# 失敗模式分析（對應論文 Table 2）
python hallucination_analysis.py --per-method react cot --tag main
```

### 2.5 環境復現

```bash
python -c "
import torch, transformers, datasets, sys
print('Python  ', sys.version.split()[0])
print('torch   ', torch.__version__, 'cuda_avail', torch.cuda.is_available())
print('transformers', transformers.__version__)
print('GPU     ', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')
"
```

完整快照見 `environment.txt`。

---

## 3. 工程結構

```
AT3_02_技术复现/
├── README.md                       本文件
├── requirements.txt                依赖清单
├── environment.txt                 运行环境快照
├── AT3_02_候选论文评估.md           第一阶段产出：10 篇候选评估与最终选定
├── run_experiment.py               主实验入口（4 条 baseline × N 题）
├── evaluate.py                     结果汇总、图表生成、配对显著性检验
├── hallucination_analysis.py       失败模式分析（对应论文 Table 2）
├── configs/                        （配置集中于 src/configs.py，此目录预留）
├── src/
│   ├── configs.py                  集中参数管理（路径可配、种子固定）
│   ├── prompts.py                  四种 prompting 格式 + 论文附录 C.1 的 6-shot 轨迹
│   ├── wikipedia_env.py            Wikipedia 环境（search / lookup / finish）
│   ├── agent.py                    ReAct 交互循环与动作解析
│   ├── llm.py                      本地 HF 模型封装（批量生成 + chat template）
│   ├── evaluation.py               答案归一化、EM / F1、Wilson 区间
│   └── data.py                     HotpotQA 加载与官方同款抽样
├── scripts/
│   └── download_data.sh            数据与模型下载
├── data/
│   ├── hotpot_qa_distractor_validation.parquet
│   └── cache/wikipedia/            Wikipedia 抓取缓存（保证实验内环境一致）
├── models/Qwen2.5-Coder-3B-Instruct/
├── results/
│   ├── metrics_main.csv            汇总指标
│   ├── per_sample_main.csv         逐题结果
│   ├── comparison_main.csv         与原论文的对照表
│   ├── hallucination_*.csv         失败模式分析
│   ├── trajectories/               完整轨迹 JSONL
│   ├── logs/                       实验日志
│   └── figures/                    图表
└── docs/
    ├── paper_analysis.md           第二阶段产出：原论文方法分析 + 流程图 + 范围界定
    └── reproduction_report.md      第五阶段产出：复现报告
```

---

## 4. 核心實現要點

### 4.1 四條 baseline 共用同一批人工軌跡

這是原論文 §3.2 的關鍵：**Act 不是另找一批樣本**，而是 ReAct 軌跡刪掉所有 thought 行。
因此 Act vs ReAct 是嚴格受控的——同一批問題、同一批動作、同一批觀測，唯一差異是中間有沒有 thought。

實現在 `src/prompts.py`：`ACT_TRAJECTORIES` 與 `REACT_THOUGHTS` 分開存放，
`build_few_shot("act")` 與 `build_few_shot("react")` 從同一份動作/觀測數據渲染。

### 4.2 環境不認識 thought

`WikipediaEnv.step("think[...]")` 返回 `"Nice thought."` 且不改變任何狀態。
這在機制上保證了論文的 $\hat{\mathcal{A}} = \mathcal{A} \cup \mathcal{L}$ 中，
thought 只能更新上下文、不能改變環境。

### 4.3 答案從 `finish[...]` 取，不從 thought 取

```python
answer = action[len("finish["):-1]      # 官方 wikienv.py 的取法
```

### 4.4 答案歸一化順序不可調換

```python
lower → remove_punc → remove_articles(a|an|the) → white_space_fix
```

論文正文與附錄均未描述此規則（僅存在於官方 `wrappers.py`），但 **EM 的可比性完全依賴於此**。
實現在 `src/evaluation.py::normalize_answer`，並有單元測試覆蓋論文中的 `Israel`/`Israeli`
標籤歧義案例。

### 4.5 步數用盡的兜底

論文只規定了 `ReAct → CoT-SC` 組合法的回退，**未規定單跑 ReAct 的兜底**；
官方代碼用 `finish[]`（空答案），該題 EM 記 0。本工程照此實現。

---

## 5. 已知偏離項（Deviations）

| # | 偏離 | 原因 | 對結果的可能影響 |
|---|---|---|---|
| D1 | 模型替換 | 8GB 顯存 + 無付費 API | 絕對 EM 遠低於原文；thought 質量下降 |
| D2 | Wikipedia 環境重建 | `en.wikipedia.org` 不可達 | 檢索質量與 2023 年不同；few-shot 中的 observation 與實測 observation 可能不一致 |
| D3 | chat template | Qwen2.5 是 instruct 模型 | 與官方 completion 格式不同；歷史軌跡作為 assistant 前綴續寫 |
| D4 | badcall 補救保留 | 官方即有 | 小模型受益更大 |
| D5 | 連續 2 次 invalid action 提前終止 | 原論文無此規則，小模型需要 | 略提高 EM（避免空答案佔比被稀釋），已列入報告 |
| D6 | 100 題而非 500 題 | 推理成本 | 小幅差異需配對檢驗才可解讀 |
| D7 | Table 2 改為規則式自動標註 | 人工標註需領域專家 | 對「事實錯誤」的靈敏度低於人工 |
| D8 | 未實現 CoT-SC 及兩條組合 | 21 樣本採樣在 3B 上方差過大 | 無法復現原文 34.2 / 35.1 的組合結論 |

---

## 6. 復現結果

完整數據見 `results/`（匯總指標、逐題結果、對照表、逐條軌跡、四張圖表）與
`docs/reproduction_report.md`。核心數據：

| 方法 | 原論文 EM（PaLM-540B） | 本次實測 EM（3B, 100 題） | 95% CI | 完成率 | 格式遵循率 | 檢索命中率 |
|---|---:|---:|---|---:|---:|---:|
| Standard | 28.7 | 16.0 | [10.1, 24.4] | 100% | 100% | — |
| CoT | 29.4 | 18.0 | [11.7, 26.7] | 100% | 100% | — |
| Act | 25.7 | 23.0 | [15.8, 32.2] | 59% | 100% | 38.2% |
| ReAct | 27.4 | 21.0 | [14.2, 30.0] | 64% | 100% | 36.3% |

**四組配對差異全部未達統計顯著**（McNemar p = 0.14–0.77，bootstrap CI 全跨 0）。
100 題不足以判定 HotpotQA 上 1–3pp 量級的差異——這是必須誠實報告的統計事實。

### 三項最重要的發現

1. **ReAct 的「格式機制」在弱模型上完全成立（格式遵循率 100%），但「收益前提」不成立**
   （檢索命中率僅 36.3%、步數用盡 36%）。**ReAct 的落地門檻在底座模型的檢索能力，而非架構。**
2. **失敗模式結構由「推理錯誤」轉向「檢索落空 + 步數用盡」**：原論文在 PaLM-540B 上步數用盡
   僅 0.84%，本次為 36–41%。
3. **自動標註無法捕捉論文的「幻覺」定義**：本次 ReAct 失敗中幻覺佔 31.6%、CoT 僅 1.2%，
   與原文（0% vs 56%）方向相反——原因是代理指標不同，非機制被推翻（詳見報告 §5.3）。

---

## 7. 授權與引用

- 本工程為課程作業，**未**經授權不推送至任何公開倉庫。
- 模型權重 Qwen2.5-Coder-3B-Instruct 遵循 Apache-2.0 授權。
- HotpotQA 數據遵循其原始授權（CC BY-SA 4.0）。
- Wikipedia 內容遵循 CC BY-SA，程式碼已在請求中設置 User-Agent 並落盤緩存。
