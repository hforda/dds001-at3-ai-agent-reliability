"""
llm.py — 本地 HuggingFace 模型封装（从零实现）

原论文使用闭源 PaLM-540B（GPT-3 text-davinci-002）。本机受 8GB 显存的限制，
改用 Qwen2.5-Coder-3B-Instruct（开源权重，本地推理，无任何付费 API 调用）。

【关键偏离声明 —— 与官方实现的差异及原因】
官方实现（text-davinci-002）是纯 completion 模型，使用如下调用：
    llm(prompt, stop=["\\nObservation {i}:"])
    -> 一次生成同时产出 Thought 与 Action，再按 "\\nAction {i}: " 切分

本实现采用 **chat template**，因为 Qwen2.5 是 instruct 模型，用纯 completion
格式会大幅掉点。具体地：
  - few-shot 示例与当前问题以 user turn 提供；
  - 历史轨迹作为 assistant turn 的前缀续写，模型续写下一段；
  - ReAct 的 Thought 与 Action 仍由**同一次生成**产出（保持官方语义），
    用 stop 序列截断并解析；
  - 解析失败时走官方的 badcall 补救（补一次调用单独生成 Action）。

温度固定为 0（贪心），对齐官方 text-davinci-002 的 temperature=0 配置。
"""
from __future__ import annotations

import logging
from typing import List, Optional, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LOGGER = logging.getLogger(__name__)


class LocalLLM:
    """本地因果语言模型的批量生成封装。"""

    def __init__(
        self,
        model_path: str,
        dtype: str = "bfloat16",
        device: str = "cuda",
        max_new_tokens: int = 160,
        temperature: float = 0.0,
        top_p: float = 1.0,
        batch_size: int = 8,
    ) -> None:
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.batch_size = batch_size

        LOGGER.info("加载模型：%s (dtype=%s, device=%s)", model_path, dtype, device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                       "float32": torch.float32}[dtype]
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch_dtype,
            device_map=device if device == "cuda" else None,
        )
        if device != "cuda":
            self.model = self.model.to(device)
        self.model.eval()
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

    # ------------------------------------------------------------------
    def generate(
        self,
        prompts: Sequence[str],
        stop_sequences: Sequence[Sequence[str]] | None = None,
        max_new_tokens: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> List[str]:
        """批量生成。stop_sequences 为每个 prompt 对应的停止序列列表。

        当 prompt 数量较多时按 batch_size 分批，避免显存溢出。
        """
        if not prompts:
            return []
        max_new = max_new_tokens or self.max_new_tokens
        if stop_sequences is None:
            stop_sequences = [[] for _ in prompts]

        bs = batch_size or self.batch_size
        if len(prompts) > bs:
            results: List[str] = []
            for i in range(0, len(prompts), bs):
                chunk_p = prompts[i:i + bs]
                chunk_s = stop_sequences[i:i + bs]
                results.extend(self._generate_chunk(chunk_p, chunk_s, max_new))
            return results
        return self._generate_chunk(list(prompts), list(stop_sequences), max_new)

    def _generate_chunk(
        self,
        prompts: List[str],
        stop_sequences: List[List[str]],
        max_new: int,
    ) -> List[str]:
        """对一批 prompt 生成。"""
        if not prompts:
            return []
        # 逐样本套用不同 stop 需逐条生成；本实现的 stop 集合在样本间一致时走批处理
        distinct = {tuple(s) for s in stop_sequences}
        if len(distinct) > 1:
            return [self._generate_one(p, s, max_new) for p, s in zip(prompts, stop_sequences)]

        stops = list(next(iter(distinct)))
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True,
                             add_special_tokens=False)
        enc = {k: v.to(self.model.device) for k, v in enc.items()}

        gen_kwargs = dict(
            max_new_tokens=max_new,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        if self.temperature and self.temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=self.temperature, top_p=self.top_p)
        else:
            gen_kwargs.update(do_sample=False)

        with torch.inference_mode():
            out = self.model.generate(**enc, **gen_kwargs)

        in_len = enc["input_ids"].shape[1]
        results = []
        for i, seq in enumerate(out):
            text = self.tokenizer.decode(seq[in_len:], skip_special_tokens=True)
            text = self._apply_stops(text, stops)
            results.append(text)
        return results

    def _generate_one(self, prompt: str, stops: Sequence[str], max_new: int) -> str:
        enc = self.tokenizer([prompt], return_tensors="pt", add_special_tokens=False)
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        gen_kwargs = dict(max_new_tokens=max_new, pad_token_id=self.tokenizer.pad_token_id,
                          eos_token_id=self.tokenizer.eos_token_id)
        if self.temperature and self.temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=self.temperature, top_p=self.top_p)
        else:
            gen_kwargs.update(do_sample=False)
        with torch.inference_mode():
            out = self.model.generate(**enc, **gen_kwargs)
        text = self.tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return self._apply_stops(text, list(stops))

    @staticmethod
    def _apply_stops(text: str, stops: Sequence[str]) -> str:
        """在最早的 stop 出现处截断。"""
        cut = len(text)
        for s in stops:
            if not s:
                continue
            idx = text.find(s)
            if idx != -1:
                cut = min(cut, idx)
        return text[:cut]

    # ------------------------------------------------------------------
    def build_chat_prompt(self, few_shot_block: str, question: str, history: str,
                          method: str) -> str:
        """把 prompt 组件组装成 chat template 的输入字符串。

        【设计决策 —— 经实测确定】
        实测发现：若在 prompt 末尾内联续写前缀（如 'Thought 1:'），3B instruct 模型
        会把该前缀当作「需要复述的内容」，输出恰好等于前缀内容后就停止，不产出 Action。
        而**不提供任何续写前缀**时，模型能一次性生成完整的
        'Thought 1: ...\\nAction 1: Search[...]' 结构，与论文附录的 few-shot 格式一致。

        因此本实现采用：prompt 只包含 few-shot 示例 + 当前问题 + 已完成的历史轨迹，
        让模型自由续写完整的下一段（Thought 与 Action 由同一次生成产出，保持官方语义）。

        - few_shot_block 作为 user turn（示例）
        - 当前问题与历史轨迹作为同一 user turn
        """
        user_parts = []
        if few_shot_block:
            user_parts.append(
                "Below are examples of the required output format. Follow the same "
                "format exactly.\n\n" + few_shot_block
            )
        q_text = f"Question: {question}"
        if history:
            q_text += "\n" + history
        user_parts.append(q_text)

        messages = [{"role": "user", "content": "\n\n".join(user_parts)}]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
