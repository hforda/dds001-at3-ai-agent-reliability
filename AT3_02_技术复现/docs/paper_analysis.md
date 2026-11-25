# 原論文方法分析：ReAct

**論文**：ReAct: Synergizing Reasoning and Acting in Language Models
**作者**：Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao（Princeton University / Google Research, Brain team）
**年份 / 會場**：2023｜ICLR 2023（In-Person Poster，top 5% paper）
**arXiv**：https://arxiv.org/abs/2210.03629 ｜項目頁：https://react-lm.github.io/
**官方代碼**：https://github.com/ysymyth/ReAct（**本次復現環境 GitHub 不可達，故全部實作從零完成**）

---

## 一、Research Problem：原論文解決什麼問題

原論文要解決的是**語言模型的「推理」與「行動」兩條研究線長期割裂**的問題。

2022 年時，LLM 的兩種能力各自被獨立研究：

| 路線 | 代表工作 | 能力 | 缺陷 |
|---|---|---|---|
| 純推理 | Chain-of-Thought (Wei et al., 2022) | 分解問題、多步推演 | **不接地**：無法查證外部事實，產生幻覺並沿推演步驟傳播錯誤 |
| 純行動 | MRKL、Toolformer、WebGPT、SayCan | 調用外部工具、API | **無推理支撐**：不知何時該調哪個工具、長程規劃薄弱、難處理例外 |

論文的問題陳述（§1）：「While large language models (LLMs) have demonstrated impressive capabilities across tasks in language understanding and interactive decision making, their abilities for reasoning (e.g., chain-of-thought prompting) and acting (e.g., action plan generation) have primarily been studied as separate topics.」

作者提出的核心命題是：**讓 LLM 以交錯（interleaved）方式生成推理軌跡與任務動作，使兩者產生協同**。推理軌跡讓模型能歸納、追蹤、更新行動計畫並處理例外；動作則讓模型能與外部資訊源互動以獲取資訊。

## 二、Core Contribution：核心創新是什麼

### 2.1 方法層面的創新（本次復現的主體）

**a) 動作空間擴充**：將動作空間從 $\mathcal{A}$ 擴充為

$$\hat{\mathcal{A}} = \mathcal{A} \cup \mathcal{L}$$

其中 $\mathcal{A}$ 是領域動作（如 `search` / `lookup` / `finish`），$\mathcal{L}$ 是**語言空間**（thought，即自然語言推理軌跡）。這是全文最核心的形式化：thought 被當作「不改變環境、只更新上下文」的動作，與環境動作享有同等的序列位置。

**b) 交錯上下文更新律**：

$$c_{t+1} = (c_t, \hat{a}_t)$$

每一步把當前生成的 $\hat{a}_t$（可能是 thought、可能是 action）追加到上下文。注意：**thought 不回傳環境**（環境狀態不變），**action 才會**（環境改變並產生 observation）。

**c) 兩種交錯密度**：論文 §3.2 明確區分：
- **稠密交錯**（知識密集任務如 HotpotQA / FEVER）：`Thought → Action → Observation → Thought → …` 逐段交替；
- **稀疏交錯**（交互決策任務如 ALFWorld / WebShop）：thought 由**模型自主決定何時插入**（「let the language model decide the asynchronous occurrence of thoughts and actions」），僅在分解目標、追蹤子目標等關鍵位置出現。

**d) 一個「刻意弱化」的檢索環境**：作者明確聲明其自建 Wikipedia API「significantly weaker than state-of-the-art lexical or neural retrievers」，目的是**迫使模型透過顯式推理來決定檢索什麼**。這是理解 ReAct 效果的關鍵——接地收益來自「模型必須主動選擇查詢詞」而非「檢索器更強」。

### 2.2 經驗層面的貢獻

- **Table 1 的七條 baseline 對照**：Standard / CoT / CoT-SC / Act / ReAct / CoT-SC→ReAct / ReAct→CoT-SC。這實質是一套**系統性消融**：同一批人工標註軌跡被反覆剝除 thought / action / observation 三種成分。
- **Table 2 的六類人工標註**：把成功與失敗拆成可比較的失敗模式分類，是全文分析價值的核心。
- **兩條回退啟發式**：`ReAct → CoT-SC`（ReAct 步數用盡則回退）與 `CoT-SC → ReAct`（CoT-SC 多數票不足 n/2 則回退）。
- **微調實驗**（附錄 B）：用 ReAct 生成的 3,000 條正確軌跡做 bootstrapping 微調，發現 ReAct/Act 方法受益於更多訓練步數，而 Standard/CoT 方法在微調後**迅速退化**。

## 三、Method Architecture：整體方法架構

```
                    ┌──────────────────────────────────────┐
   Task instance ──▶│        Prompt 構造（few-shot）        │
   (question/claim)│  Standard | CoT | Act | ReAct          │
                    │  6 條人工標註軌跡（HotpotQA）         │
                    └───────────────┬──────────────────────┘
                                    │  c_0（軌跡歷史）
                                    ▼
        ┌───────────────────────────────────────────────────┐
        │            LLM（frozen PaLM-540B）                 │
        │   輸入：c_t + "Thought {t}:"（或 "Action {t}:"） │
        │   輸出：thought（可選）+ action                    │
        │   停止符："\nObservation {t}:"                    │
        └───────────────┬───────────────────────────────────┘
                        │  â_t ∈ Â = A ∪ L
                        ▼
              ┌──────────────────┐
              │  動作解析        │──  action[0].lower() + action[1:]
              └────────┬─────────┘
                       ▼
        ┌───────────────────────────────────────┐
        │        Wikipedia API（環境）           │
        │  search[entity] / lookup[string] /     │
        │  finish[answer]                        │
        │  （環境只接受 A 中的動作，不接受 thought）│
        └────────┬──────────────────────────────┘
                 │  observation_t（純文字回饋）
                 ▼
              c_{t+1} = (c_t, â_t, observation_t)
                       │
                       ├─ 若 action = finish[answer] → done，reward=1
                       └─ 否則 → 回到 LLM，t += 1
                              （t 達上限 7 步 → 強制 finish[] 空答案）
```

### 3.1 環境與模型的職責邊界（架構關鍵）

論文的一個精妙之處在於**職責分離**：模型負責「生成 thought 與 action」，環境負責「解釋 action 並回傳 observation」，而環境**完全不認識 thought**。官方 `wikienv.py` 中，環境僅接受 `search` / `lookup` / `finish` 三種動作，若收到 `think[...]` 會回 `"Nice thought."`——這個設計使得 thought 在機制上**不可能**改變環境，強化了 §2.1(b) 的性質。

## 四、Algorithm：核心算法與執行流程

### 4.1 論文 §3.2 的完整交互循環

```
輸入：q（問題）、prompt P（few-shot 示例）
c ← P
for t = 1 to 7:                      # HotpotQA 7 步；FEVER 論文為 5 步
    â_t ← LLM(c + "Thought {t}:")    # stop = ["\nObservation {t}:"]
    若無法切分出 "\nAction {t}: "：   # badcall
        thought ← â_t 的第一行
        action ← LLM(c + "Thought {t}: {thought}\nAction {t}:")   # stop=["\n"]
    否則：
        (thought, action) ← 切分
    observation ← Env.execute(action)
    c ← c + "Thought {t}: {thought}\nAction {t}: {action}\nObservation {t}: {observation}\n"
    若 action = finish[...]: break
若未終止：
    observation ← Env.execute("finish[]")   # 空答案，EM 記 0
```

### 4.2 官方實作中一個易被忽略的細節：`badcall` 處理

當模型輸出無法按 `\nAction {t}: ` 切分時，官方**不是直接放棄**，而是補一次 LLM 呼叫單獨生成 action。這提高了格式遵循失敗時的存活率。此機制在本次復現中保留（因為小模型格式遵循能力弱，這一補救對結果影響顯著，須在報告中聲明）。

### 4.3 七條 baseline 的構造（系統性消融）

論文 §3.2 原文說明所有 baseline 都由**同一批 ReAct 人工軌跡**剝除成分得到：

| Baseline | 構造方式（論文明示） | 輸出格式 |
|---|---|---|
| **Standard** | 「removes all thoughts, actions, observations in ReAct trajectories」 | `Question:` / `Answer:` |
| **CoT** | 「removes actions and observations and serve as a reasoning-only baseline」 | `Question:` / `Thought:` / `Answer:` |
| **CoT-SC** | 採樣 **21** 條 CoT 軌跡，temperature **0.7**，多數票 | 同 CoT |
| **Act** | 「removes thoughts in ReAct trajectories, loosely resembling how WebGPT interacts with the Internet」 | `Question:` / `Action N:` / `Observation N:` |
| **ReAct** | 完整軌跡 | `Question:` / `Thought N:` / `Action N:` / `Observation N:` |
| **CoT-SC → ReAct** | CoT-SC 多數票出現次數 < n/2 → 回退 ReAct | 混合 |
| **ReAct → CoT-SC** | ReAct 步數用盡 → 回退 CoT-SC | 混合 |

**關鍵**：Act 不是另找一批樣本重新標註，而是**同一批軌跡刪掉 thought 行**。這使得 Act 與 ReAct 的比較是嚴格受控的——同一批問題、同一批動作、同一批觀測，唯一差異是中間有沒有 thought。

### 4.4 環境（Wikipedia API）的精確定義

論文 §3.1 原文：

> (1) `search[entity]`, which returns **the first 5 sentences** from the corresponding entity wiki page if it exists, or else suggests **top-5 similar entities** from the Wikipedia search engine,
> (2) `lookup[string]`, which would return **the next sentence in the page containing string**, simulating Ctrl+F functionality on the browser.
> (3) `finish[answer]`, which would finish the current task with answer.

官方 `wikienv.py` 補充了論文未寫的行為細節（本次復現照此實現）：

| 行為 | 官方實現細節 |
|---|---|
| `search` 成功 | 頁面按段落切 → 按句切 → **取前 5 句以空格拼接**回傳 |
| `search` 失敗 | `f"Could not find {entity}. Similar: {self.result_titles[:5]}."` —— 恰好 5 個候選 |
| `search` 重定向 | 若頁面含 "may refer to:"，自動遞迴搜尋 `"[" + entity + "]"` |
| `lookup` 回傳 | `f"(Result {cnt+1} / {len(list)}) " + sentence` —— **帶進度前綴** |
| `lookup` 越界 | `"No more results.\n"` |
| `lookup` 游標 | 同一 keyword 連續 lookup **遞增游標**回傳第 2、3 個結果；換 keyword 則重置 |
| `finish` | `f"Episode finished, reward = {reward}\n"`，`done=True` |
| 非法動作 | `f"Invalid action: {action}"` |
| `think[...]` | `"Nice thought."`（**保留為合法動作但不改變環境**） |

### 4.5 答案解析與歸一化

| 格式 | 答案來源 |
|---|---|
| ReAct / Act | `finish[<answer>]` **方括號內的原始字串** |
| CoT / Standard | `Answer:` 行之後的內容 |
| 步數用盡 | `finish[]` → 空串 → 歸一化後為空 → 與任何 GT 皆不等 → EM = 0 |

**歸一化規則（官方 `wrappers.py` `normalize_answer`，論文正文未寫但 EM 可比性必需）**，順序嚴格為：

```
lower → remove_punc → remove_articles → white_space_fix
```

- `remove_punc`：刪除全部 `string.punctuation`（ASCII 標點；**不含 unicode 標點**，如 `–`、彎引號 `'` 不會被刪）
- `remove_articles`：`re.sub(r"\b(a|an|the)\b", " ", text)`
- `white_space_fix`：`" ".join(text.split())`

## 五、Input / Output

| 項目 | 內容 |
|---|---|
| **Input** | HotpotQA：問題字串 q（+ gold answer 僅用於評分，不進 prompt）<br>FEVER：claim 字串 c（+ 三類標籤） |
| **中間態** | 上下文 c_t = few-shot 示例 + 已生成的 (thought, action, observation) 序列 |
| **Output** | 單一答案字串 a（從 `finish[...]` 或 `Answer:` 解析） |
| **中間觀測** | 每次環境互動回傳一段純文字 observation（search 的前 5 句 / lookup 的 `(Result i / n) 句子` / finish 的 reward 字串） |

## 六、Data：數據集與劃分

| 數據集 | 規模 | 用途 | 官方取得方式 |
|---|---|---|---|
| **HotpotQA distractor dev** | **7,405** 題 | 主評測集 | 官方 repo `data/hotpot_dev_v1_simplified.json` |
| HotpotQA train | 43,231 題 | **選 few-shot 示例的來源**（6 題） | 官方 repo `data/hotpot_train_v1.1_simplified.json` |
| **FEVER dev (`paper_dev`)** | **9,999** 題（三類各 3,333） | FEVER 主評測集 | 官方 repo `data/paper_dev.jsonl` |
| ALFWorld | 134 個未見驗證實例 | 決策任務（**本次不複現**） | ALFRED |

**評測子集抽樣（官方代碼）**：
```python
idxs = list(range(7405))
random.Random(233).shuffle(idxs)
idxs[:500]          # 取前 500 題
```
即 **seed = 233，隨機抽 500 題**。

**few-shot 示例來源**：HotpotQA 6 題、FEVER 3 題，**從訓練集隨機選取後人工撰寫 ReAct 格式軌跡**（含 thought、action、observation 三部分，observation 由當時的 Wikipedia 實時抓取）。論文腳註 2：「We find more examples do not improve performance.」

**thought 的六種類型**（論文 §2 明列，本次複現撰寫示例時的依據）：
1. 分解問題（「I need to search x, find y, then find z」）
2. 從觀測抽取資訊（「x was started in 1844」）
3. 常識推理（「x is not y, so z must instead be…」）
4. 算術推理（「1844 < 1989」）
5. 引導檢索改寫（「maybe I can search/look up x instead」）
6. 綜合最終答案（「…so the answer is x」）

## 七、Metrics：評價指標

| 指標 | 定義 | 備註 |
|---|---|---|
| **EM（Exact Match）** | 歸一化後答案字串完全相等 | 論文 Table 1 對 HotpotQA **只報 EM，未報 F1** |
| **F1** | SQuAD 式 token 重疊 F1 | 官方 `wrappers.py` 額外提供；本次複現一併輸出 |
| **FEVER Acc** | 三類標籤完全匹配 | FEVER 的 em ≡ f1 ≡ accuracy |
| **完成率（完成度）** | 在步數上限內產出合法 `finish[]` 的比例 | **本次複現新增的診斷指標**，用於解釋小模型分數偏低的原因 |
| **平均步數** | 實際消耗的環境互動步數 | 本次複現新增 |
| **格式遵循率** | 生成可被解析為合法動作的比例 | 本次複現新增 |
| **幻觉率分類** | 論文 Table 2 的六類人工標註 | 本次複現以 LLM-as-judge + 規則輔助近似，並報告與人工標註的差異 |

## 八、Baseline：原論文採用的對照方法

原論文 Table 1 共 8 條（含 2 條組合方法與 1 條 supervised 上界）：

| 方法 | HotpotQA (EM) | Fever (Acc) | 備註 |
|---|---:|---:|---|
| Standard | 28.7 | 57.1 | |
| CoT | 29.4 | 56.3 | |
| CoT-SC | 33.4 | 60.4 | 21 樣本，temp 0.7 |
| Act | 25.7 | 58.9 | |
| **ReAct** | **27.4** | **60.9** | |
| CoT-SC → ReAct | 34.2 | 64.6 | 組合 |
| ReAct → CoT-SC | **35.1** | 62.0 | 組合 |
| Supervised SoTA | 67.5 | 89.5 | 上界，非 prompting 方法 |

**附錄 A.1 Table 5（GPT-3 text-davinci-002，貪心解碼，500 題 HotpotQA 子集）**：HotpotQA EM **30.8**、ALFWorld SR **78.4**；對照 PaLM-540B 的 29.4 / 70.9。

## 九、Key Results：最重要的實驗結果

### 9.1 主結果（Table 1，PaLM-540B）

除上表外，三點最關鍵：

1. **HotpotQA 上純 ReAct（27.4）低於純 CoT（29.4）**。ReAct 的優勢需要與 CoT-SC 結合才顯現（ReAct → CoT-SC 達 35.1）。這是本次複現必須同時實現 CoT 與 Act 兩條 baseline 的原因——否則無法判定交錯格式在何種條件下真正有效。
2. **FEVER 上 ReAct（60.9）高於 CoT（56.3）與 Standard（57.1）**。論文 §3.3 解釋：FEVER 的 SUPPORTS/REFUTES 兩類 claim 往往只差一點，**必須實際檢索才能取得準確且最新的知識**——這是結構性的差異，不依賴長鏈推理能力。
3. **Act 在 HotpotQA（25.7）低於 Standard（28.7）**，但在 FEVER（58.9）高於 Standard（57.1）與 CoT（56.3）。

### 9.2 失敗模式分析（Table 2）

**採樣協議**（§3.3）：從 ReAct 與 CoT 各隨機抽取 **50 條正確 + 50 條錯誤**軌跡（以 EM 判定正誤），**合計 200 條**，由人工標註。

| Type | 定義 | **ReAct** | **CoT** |
|---|---|---:|---:|
| Success – True positive | 正確的推理軌跡與事實 | 94% | 86% |
| Success – **False positive** | **幻覺的推理軌跡或事實** | **6%** | **14%** |
| Failure – Reasoning error | 錯誤推理（含無法從重複步驟中恢復） | 47% | 16% |
| Failure – Search result error | 檢索回傳空結果或不含有用資訊 | **23%** | – |
| Failure – **Hallucination** | **幻覺的推理軌跡或事實** | **0%** | **56%** |
| Failure – Label ambiguity | 預測正確但未精確匹配標籤 | 29% | 28% |

**論文的機制解讀**：
- **A)** 幻覺是 CoT 的嚴重問題（成功案例假陽性 14% vs 6%；失敗案例幻覺佔 56%），ReAct 因可接入外部知識庫而更 grounded、more trustworthy。
- **B)** 但結構約束也降低了推理靈活性，導致 ReAct 的 reasoning error 率**高於** CoT（47% vs 16%）。腳註 6 指出這類失敗的具體形態：**模型反覆生成先前的 thought 與 action 形成循環**，作者歸因於**次優的貪心解碼**（認為 beam search 可改善）。
- **C)** 對 ReAct 而言，**成功檢索到有用知識至關重要**：非資訊性檢索佔錯誤案例的 **23%**，會讓模型推理脫軌且難以恢復。

### 9.3 微調實驗（附錄 B.1）

用 ReAct 生成的 3,000 條正確軌跡做 bootstrapping 微調 PaLM-8B/62B：ReAct 與 Act 方法受益於更多訓練步數與資料，而 **Standard 與 CoT 方法在微調後迅速退化**。

### 9.4 數據過時性（附錄 A.2）

論文自己發現 HotpotQA 部分問題的**答案標籤已過時**（hotel 高度案例，附錄 A.2 + Figure 4）。這對本次復現是重要的現實約束。

## 十、Limitations：原論文的局限性

| # | 局限 | 出處 |
|---|---|---|
| L1 | **未提供獨立的 Limitations 章節**；限制散見於 §4 討論與倫理聲明 | 全文 |
| L2 | **推理靈活性下降**：交錯結構雖提升接地性但限制推理，reasoning error 率 47% vs CoT 的 16% | §3.3-B |
| L3 | **重複循環模式**：模型反覆生成先前 thought/action，作者歸因於貪心解碼 | §3.3-B 腳註 6 |
| L4 | **檢索失效難以恢復**：非資訊性檢索佔錯誤 23%，使推理脫軌 | §3.3-C |
| L5 | **檢索能力刻意弱化**：自建 API 遠弱於 SOTA 檢索器，結論外推性有限 | §3.1 |
| L6 | **大型動作空間的 prompt 成本**：demonstrations 佔用輸入長度 | §4 |
| L7 | **步數上限是經驗設定**：7 步（HotpotQA）/ 5 步（FEVER），「更多步數不會更好」但未給出系統性分析 | §3.2 + 腳註 3 |
| L8 | **資料過時**：HotpotQA 部分答案標籤已過時 | 附錄 A.2 |
| L9 | **未報告 Table 1 的評測題數**（附錄 A.1 的 500 題僅針對 GPT-3 實驗） | Table 1 vs 附錄 A.1 |
| L10 | **倫理風險**：接入外部動作空間帶來隱私與有害動作風險；實驗限於 Wikipedia 與 WebShop | 倫理聲明 |

## 十一、核心方法流程圖

```
┌──────────────────────────────────────────────────────────────────────────┐
│ INPUT：HotpotQA 問題 q（gold answer 僅供評分，不進 prompt）              │
└───────────────────────────────────────┬──────────────────────────────────┘
                                        ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ MODULE 1：Prompt 構造（6-shot 人工標註軌跡，四格式共用同一批示例）      │
│  Standard: Q/A │ CoT: Q/Thought/A │ Act: Q/Action/Obs │ ReAct: Q/Th/Act/O │
│  ★ Act = ReAct 刪除所有 thought 行（嚴格受控消融）                       │
└───────────────────────────────────────┬──────────────────────────────────┘
                                        ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ MODULE 2：LLM 一步生成（frozen；stop = "\nObservation {t}:"）             │
│  輸出 = thought（僅 ReAct/CoT）× action                                  │
│  格式失敗 → badcall 補一次呼叫單獨生成 action                            │
└───────────────────────────────────────┬──────────────────────────────────┘
                                        ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ MODULE 3：動作解析（action[0].lower() + action[1:]）                      │
│  解析出 search[entity] / lookup[string] / finish[answer]                │
│  ★ 環境不認識 thought → thought 在機制上不可能改變環境                   │
└───────────────────────────────────────┬──────────────────────────────────┘
                                        ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ MODULE 4：Wikipedia 環境（★ 核心創新所在）                              │
│  search[e]  → 頁面前 5 句；查無則 "Could not find e. Similar: [5 項]"  │
│  lookup[s]  → "(Result i / n) 下一句"；游標式遞增；越界 "No more results"│
│  finish[a]  → done=True，回傳 reward 字串                              │
└───────────────────────────────────────┬──────────────────────────────────┘
                                        ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ MODULE 5：上下文更新  c_{t+1} = (c_t, â_t, observation_t)                │
│  ★ observation 回填是接地收益的來源；無觀測即無接地                     │
└───────────────────────────────────────┬──────────────────────────────────┘
                          ┌─────────────┴─────────────┐
                    action = finish[?]           action ≠ finish[?]
                          ▼                             ▼
                  ┌───────────────┐         ┌──────────────────────┐
                  │ done, 解析答案 │         │ t += 1；t > 7 →      │
                  │ finish[...]   │         │ 強制 finish[] 空答案 │
                  └───────┬───────┘         │ （EM = 0）            │
                          │                 └──────────┬───────────┘
                          └────────────┬───────────────┘
                                       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ MODULE 6：評價  normalize_answer(lower→punc→articles→ws) → EM / F1       │
│  輔助診斷：完成率、平均步數、格式遵循率、檢索成功率、幻覺率分類          │
└──────────────────────────────────────────────────────────────────────────┘
```

### 11.1 公式的實際作用（非形式化抄寫）

| 公式 | 實際作用 |
|---|---|
| $\hat{\mathcal{A}} = \mathcal{A} \cup \mathcal{L}$ | 把「想」與「做」放進同一個序列位置，使模型能在同一個生成步驟裡自由切換。這是 ReAct 與純 CoT、純 ReAct（WebGPT 式）的**根本區別**——後兩者的序列結構是硬編碼的 |
| $c_{t+1} = (c_t, \hat{a}_t)$ | 上下文只追加不重寫，使完整軌跡成為模型下一步的全部依據。**這是「接地」的機制來源**：observation 進入了上下文，模型下一個 thought 就有了事實錨點 |
| `pass^1` 等效的 EM | 歸一化順序不可調換（lower→punc→article→ws），否則 EM 與原文不可比 |

## 十二、本次複現的範圍界定

| 原論文內容 | 本次複現內容 | 簡化或省略 | 原因 |
|---|---|---|---|
| **交錯推理-行動循環** | ✅ 完整實現 | — | 核心演算法，不允許刪除 |
| **動作空間擴充 Â = A ∪ L** | ✅ 完整實現 | — | 核心機制，不允許刪除 |
| **三動作 Wikipedia 環境** | ✅ 完整實現 | 改用 `api.wikimedia.org` 重建（主站被阻斷） | 語義等價：search 返回前 5 句 + top-5 相似實體；lookup 返回含目標字串的句子 + `(Result i / n)` 前綴 |
| **6-shot 人工標註軌跡（HotpotQA）** | ✅ 完整實現（依附錄 C.1 逐字） | — | 核心機制 |
| **Act = ReAct 刪 thought 的受控消融** | ✅ 完整實現 | — | 核心機制 |
| **Standard / CoT 兩條 baseline** | ✅ 完整實現 | — | 對照必需 |
| **CoT-SC（21 樣本, temp 0.7）** | ❌ 省略 | 不做 | 21 倍採樣成本，在 8GB 顯存的 3B 模型上採樣方差會淹沒訊號；且原論文的組合結論主要依賴 PaLM-540B |
| **CoT-SC→ReAct / ReAct→CoT-SC 組合** | ❌ 省略 | 不做 | 同上，依賴 CoT-SC |
| **FEVER 任務** | ❌ 省略 | 不做 | 本次以 HotpotQA 為主，FEVER 需另一套 claim 資料與三類標籤；§9.1 顯示 FEVER 是 ReAct 相對優勢最可信的場景，列為後續工作 |
| **ALFWorld / WebShop** | ❌ 省略 | 不做 | 環境安裝成本高，且 success rate 對格式更敏感 |
| **PaLM-540B 主模型** | ⚠️ 換用 Qwen2.5-Coder-3B-Instruct | 模型規模 | 8GB 顯存上限，無付費 API 額度 |
| **Table 1 的 8 條 baseline** | ⚠️ 實現 4 條 | 減半 | CoT-SC 及其兩條組合依賴 21 樣本採樣 |
| **HotpotQA 評測 500 題** | ⚠️ 縮至 100 題 | 縮小 | 3B 模型在 8GB 顯存下的推理速度；每題需 ≤7 步 × 4 baseline 的生成量 |
| **步數上限 7 步** | ✅ 保留 7 步 | — | 核心設定，不改動 |
| **Thought+Action 單次生成** | ⚠️ 保留官方單次生成，失敗時走官方 badcall 補救 | — | 偏離項，須聲明 |
| **答案歸一化規則** | ✅ 完整實現（照官方 `normalize_answer`） | — | EM 可比性必需 |
| **Table 2 人工標註 200 條** | ⚠️ 改為 LLM-as-judge 六類分類（規則輔助） | 自動化 | 人工標註需領域專家投入；本次用 judge 模型近似，並報告與人工標註口徑的差異 |
| **微調實驗（3,000 條軌跡）** | ❌ 省略 | 不做 | ReAct 主結論建立在 frozen LLM 的 prompting 上；微調需 GPU 訓練，8GB 顯存不可行 |
| **官方代碼** | ❌ 不使用 | — | GitHub 不可達；且任務書明令不得把下載運行官方代碼包裝為獨立實現 |

### 12.1 明確聲明的偏離項（Deviations）

1. **模型替換**：PaLM-540B → Qwen2.5-Coder-3B-Instruct（3B）。**絕對 EM 必然顯著低於原文 27.4 / 29.4 / 25.7 / 28.7**，這是預期內的規模效應，非複現失敗。
2. **檢索環境重建**：Wikipedia 主站被網路環境阻斷，改用 `api.wikimedia.org`。檢索質量與原論文自建的簡易 API 未必相同（且 2026 年的頁面內容與 2023 年 few-shot 示例記錄的 observation 必然有差異——這本身會影響模型行為，須在報告中量化）。
3. **採樣策略**：採貪心解碼（temperature=0），對齊官方 GPT-3 配置。
4. **Table 2 自動化**：以 LLM-as-judge 近似人工標註，結果與原文 6% / 14% / 0% / 56% 的比較須視為**方向性對照**而非數值對等。
5. **評測題數**：100 題（原 500 題）。100 題上 EM 的 95% 信賴區間寬度約 ±9 個百分點，這是解讀相對差異時必須納入的統計約束。

---

**下一步**：依本分析界定範圍，開始第三階段的代碼實現。
