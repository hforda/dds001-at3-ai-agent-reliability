# 第一轮实验（iteration_1）结论标记

> **结论：程序形式验证通过，但经人工审查发现 citation 语义支持不足，不能作为最终提交版本。**

## 本目录内容

`results/iteration_1/` 完整保存第一轮 `max_steps=3` / `max_steps=6` 的原始执行证据，
**未经删除、篡改或重跑**。所有文件与首轮运行时的字节内容一致（已用 SHA256 校验）。

```
results/iteration_1/
├── maxsteps3/                max_steps=3 组的完整输出
│   ├── final-brief.md        （未生成简报）
│   ├── trajectory.jsonl      逐步决策记录
│   ├── tool-results.jsonl    每次工具调用的完整记录
│   ├── evaluation.csv        汇总指标 + 八项 gate 结果
│   ├── provenance.json       真实 provenance
│   └── validator.json        final-state validator 输出（FAIL 3/8）
├── maxsteps6/                max_steps=6 组的完整输出（PASS 8/8）
├── cited_sources/            第一轮引用的三篇来源完整记录
├── comparison.csv            两组逐指标对照
├── comparison_gates.csv      两组逐 gate 对照
├── validator_maxsteps3.json  max_steps=3 validator 输出副本
├── validator_maxsteps6.json  max_steps=6 validator 输出副本
├── final-brief.md            第一轮最终简报副本
├── trajectory.jsonl          第一轮最终轨迹副本
├── tool-results.jsonl        第一轮最终工具结果副本
├── evaluation.csv            第一轮最终汇总副本
└── provenance.json           第一轮 provenance 副本
```

## 为什么不能作为最终提交版本

第一轮在**程序层面**全部通过：Agent 真实调用工具循环、validator 八项 gate 判定 PASS、
citation 的 doc_id 与句号在来源包中均可定位。但人工逐条审查后发现：

| 编号 | 问题 | 性质 |
|---|---|---|
| 1 | 引用由工具侧的**词汇重叠自动对齐**生成（`_autofill_sentence_ids`），模型并未真正把某句原文抄进 bullet | 引用非模型自主产出 |
| 2 | 当模型只写 2 个来源时，工具**自动补足第三篇来源**的 marker | 引用来源非模型自主选择 |
| 3 | 「引用句与简报陈述的语义支持」检查最初为硬门禁，后被降级为诊断指标（`semantic_alignment_rate`） | 门禁强度被事后放宽 |
| 4 | 引用来源之一 `2301.11659v2`（Linear Algebra / Tensor Code to Hardware Accelerators）与任务主题 **LLM Evaluation 无直接关系** | 来源选择错误 |
| 5 | Prompt 示例中直接写入真实 doc_id，暗示模型「应该」读哪些文档 | 污染 Agent 自主决策 |
| 6 | `read_source` 无候选 allowlist，模型可读取未出现在搜索结果中的文档 | 工具层约束缺失 |

第 1、2 项尤其关键：第一轮所谓「6 个可回查引用」中，有一部分是**工具替模型匹配出来的**，
而非模型自己从读到的原文中选取。这属于「把生成简报等同于完成 Agent 任务」与
「把工具调用次数等同于有效工作」的边界模糊，**不符合教师对 citation 真实性的要求**。

## 第二轮的修正方向

第二轮（见上级目录的 `results/run_maxsteps3/`、`results/run_maxsteps6/`）针对上述六点逐项修正：

1. 删除所有词汇重叠自动对齐与自动补足来源的逻辑；
2. 要求模型在 bullet 中**逐字写出**所依据的来源原句；
3. 引用句与论断的对应关系恢复为**硬门禁**（逐字核验，不接受仅一个共同词）；
4. 重新选择与 LLM Evaluation 主题真正相关的三篇来源；
5. Prompt 中移除真实 doc_id，改用占位符 + 候选 allowlist；
6. 工具层强制 `read_source` 只能读取 `search_sources` 返回的候选。

## 引用纪律

本目录内容仅作为**历史证据**保留，供审计与对照，不得作为最终提交版本的结论依据。
最终提交版本以 `results/run_maxsteps3/`、`results/run_maxsteps6/` 及根目录各交付文件为准。
