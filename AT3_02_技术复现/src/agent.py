"""
agent.py — ReAct 的四种 prompting 方法与交互循环（从零实现）

本模块实现论文 §3.2 的算法结构：

    for t = 1..7:
        a_hat_t <- LLM(c_t + "Thought {t}:")     # stop = "\nObservation {t}:"
        若解析失败（badcall）:
            thought <- a_hat_t 首行
            action  <- LLM(c_t + "Thought {t}: {thought}\nAction {t}:")   # stop=["\n"]
        observation <- Env.execute(action)
        c_{t+1} <- c_t + "Thought {t}: {thought}\nAction {t}: {action}\nObservation {t}: {obs}"
        若 action = finish[...]: break
    若未终止: Env.execute("finish[]")   # 空答案，EM = 0

Standard 与 CoT 不与环境交互，只需一次生成。
Act 与 ReAct 与 Wikipedia 环境交互。

【与官方实现的关键偏离及原因】
1. 使用 chat template（Qwen2.5 为 instruct 模型，纯 completion 格式会掉点）。
2. 历史轨迹作为 assistant turn 的前缀续写；Thought 与 Action 仍由同一次生成产出，
   保持官方语义。解析失败时走官方 badcall 补救。
3. 加入连续 invalid action 的提前终止（原论文无此规则，小模型需要，
   已作为偏离项在报告中声明）。
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from . import prompts as P
from .evaluation import compute_metrics
from .llm import LocalLLM
from .wikipedia_env import WikipediaEnv

LOGGER = logging.getLogger(__name__)

# 合法动作：search / lookup / finish（think 由环境处理但不改变状态）
_ACTION_RE = re.compile(r"^\s*(search|lookup|finish)\s*\[(.*)\]\s*$", re.IGNORECASE | re.DOTALL)
# 宽松匹配：允许缺少右括号、括号形式不同、JSON 风格 {"action": "Search[...]"}
_LOOSE_ACTION_RE = re.compile(
    r"(?:^|[\s\"'`{,])(?:Action\s*\d*\s*[:：]?\s*)?[\"']?\s*"
    r"(search|lookup|finish)\s*[\(\[\{]\s*[\"']?",
    re.IGNORECASE,
)
# 括号配对扫描：从动作名后的括号开始，取到配对的闭括号（支持嵌套，如 Search[High Plains (United States)]）
_BRACKET_PAIRS = {"(": ")", "[": "]", "{": "}"}


def _scan_balanced(text: str, start: int) -> str:
    """从 text[start] 处的括号开始扫描，返回括号内的内容（支持嵌套）。

    若括号未配对，则返回到行尾的内容。
    """
    opener = text[start]
    closer = _BRACKET_PAIRS.get(opener)
    if closer is None:
        return ""
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start + 1:i]
    # 未配对：取到行尾
    return text[start + 1:].split("\n")[0]


@dataclass
class Trajectory:
    """一条完整的交互轨迹及其评测结果。"""
    qid: str
    question: str
    gold_answer: str
    method: str
    prediction: str = ""
    steps: int = 0
    n_actions: int = 0
    n_invalid_actions: int = 0
    n_badcalls: int = 0
    completed: bool = False          # 是否在步数内产出合法 finish
    format_ok: bool = False          # 是否成功解析出合法动作
    search_attempts: int = 0
    search_hits: int = 0             # search 返回了实质内容（非 "Could not find"）
    lookups: int = 0
    n_repeated_actions: int = 0      # 与此前完全相同的动作次数（论文脚注 6 的循环模式）
    history: str = ""
    raw_outputs: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    error: str = ""


def parse_action(text: str) -> Tuple[str, str]:
    """从模型输出中解析 (action_name, action_arg)。

    动作名统一规范化为小写（search / lookup / finish）。
    官方实现用 `action[0].lower() + action[1:]`（只降首字母），但小模型常整体输出
    小写或全大写，故此处做完全规范化，保证环境能正确识别。

    返回 ('','') 表示解析失败（触发 badcall 补救）。
    """
    if not text:
        return "", ""
    m = _ACTION_RE.match(text.strip())
    if m:
        # 严格匹配成功：括号内容可能仍含嵌套（Search[High Plains (United States)]），
        # 用平衡扫描复核
        name = m.group(1).lower()
        arg = m.group(2)
        # 若严格匹配的尾部不是 ']'，说明有嵌套括号，改用平衡扫描
        idx = text.strip().find(m.group(1), 0)
        brace = text.strip().find("[", idx) if idx != -1 else -1
        if brace != -1:
            scanned = _scan_balanced(text.strip(), brace)
            if scanned:
                return name, scanned.strip()
        return name, arg.strip()

    # 宽松匹配：取最后一个动作（模型可能重复输出）
    matches = list(_LOOSE_ACTION_RE.finditer(text))
    if matches:
        last = matches[-1]
        name = last.group(1).lower()
        brace = last.end() - 1  # 匹配串以开括号结尾
        arg = _scan_balanced(text, brace)
        return name, arg.strip().strip("\"'")
    return "", ""


def extract_finish_answer(text: str) -> Optional[str]:
    """从 finish[...] 中取答案（官方 wikienv 的取法）。"""
    m = re.search(r"finish\s*\[(.*?)\]", text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    return None


def extract_answer_line(text: str) -> str:
    """从 CoT / Standard 输出中取 Answer: 之后的内容。"""
    if not text:
        return ""
    m = re.search(r"Answer\s*[:：]\s*(.*)", text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip().split("\n")[0].strip()
    return text.strip().split("\n")[0].strip()


class ReActAgent:
    """ReAct 复现的主控类。"""

    def __init__(
        self,
        llm: LocalLLM,
        env: Optional[WikipediaEnv] = None,
        max_steps: int = 7,
        max_consecutive_invalid: int = 2,
        max_generation_retries: int = 2,
    ) -> None:
        self.llm = llm
        self.env = env
        self.max_steps = max_steps
        self.max_consecutive_invalid = max_consecutive_invalid
        self.max_generation_retries = max_generation_retries

    # ------------------------------------------------------------------
    # 单次生成（Standard / CoT）
    # ------------------------------------------------------------------
    def run_direct(self, method: str, qid: str, question: str, gold: str) -> Trajectory:
        """Standard 与 CoT：不与环境交互，一次生成即作答。"""
        traj = Trajectory(qid=qid, question=question, gold_answer=gold, method=method)
        few_shot = P.build_few_shot(method)
        prompt = self.llm.build_chat_prompt(few_shot, question, "", method)
        try:
            out = self.llm.generate([prompt], stop_sequences=[["\n\nQuestion:"]])[0]
        except Exception as exc:  # noqa: BLE001
            traj.error = f"generation_failed: {exc}"
            LOGGER.error("[%s] 生成失败 qid=%s: %s", method, qid, exc)
            out = ""
        traj.raw_outputs.append(out)
        traj.steps = 1
        traj.completed = True
        traj.format_ok = bool(out.strip())
        traj.prediction = extract_answer_line(out)
        traj.metrics = compute_metrics(traj.prediction, gold)
        return traj

    # ------------------------------------------------------------------
    # 交互循环（Act / ReAct）
    # ------------------------------------------------------------------
    def run_interactive(self, method: str, qid: str, question: str, gold: str) -> Trajectory:
        """Act 与 ReAct：与 Wikipedia 环境交互直到 finish 或步数用尽。

        注意：环境持有逐回合状态（当前页面、lookup 游标），故每题开始前必须
        重置，否则上一题的页面会污染本题的 lookup 语义。
        """
        if self.env is not None:
            self.env.reset_episode()
        traj = Trajectory(qid=qid, question=question, gold_answer=gold, method=method)
        few_shot = P.build_few_shot(method)
        history_lines: List[str] = []
        consecutive_invalid = 0
        used_actions: List[str] = []      # 用于检测重复动作（论文脚注 6 的循环模式）

        for step in range(1, self.max_steps + 1):
            history = "\n".join(history_lines)
            prompt = self._build_step_prompt(few_shot, question, history, method, step)

            # 不設 stop 序列：讓模型自然生成完整的一步（Thought + Action）。
            # 實測 3B 模型在 stop=["\\nObservation {i}:"] 下會提前截斷在 Action 之前，
            # 而無 stop 時能產出與論文 few-shot 一致的 'Thought i / Action i' 結構。
            stops: List[str] = []

            try:
                out = self.llm.generate([prompt], stop_sequences=[stops])[0]
            except Exception as exc:  # noqa: BLE001
                traj.error = f"generation_failed: {exc}"
                LOGGER.error("[%s] 生成失败 qid=%s step=%d: %s", method, qid, step, exc)
                break

            traj.raw_outputs.append(out)
            traj.steps = step

            # ---- 解析：取模型输出中的**第一段** Thought+Action ----
            # 模型可能一次生成多步（Thought1/Action1/Observation1/Thought2/...），
            # 但 Observation 必须由环境真实执行得到，因此只取第一段，
            # 后续步骤交回下一轮由环境反馈驱动。
            action_name, action_arg, thought = self._parse_first_step(out, method, step)

            if not action_name:
                # badcall 补救：单独生成 Action
                action_name, action_arg = self._retry_action(
                    few_shot, question, history, thought, step, method, traj
                )

            if not action_name:
                # 两次补救仍失败：记为 invalid action
                traj.n_invalid_actions += 1
                consecutive_invalid += 1
                obs = f"Invalid action: {out.strip()[:120]}"
                history_lines.append(self._format_step(method, step, thought, "", obs))
                if consecutive_invalid >= self.max_consecutive_invalid:
                    LOGGER.debug("[%s] 连续 invalid，提前终止 qid=%s", method, qid)
                    break
                continue

            consecutive_invalid = 0
            traj.n_actions += 1
            traj.format_ok = True

            # ---- 重复动作检测（论文脚注 6 描述的循环失败模式）----
            action_key = f"{action_name}[{action_arg.lower()}]"
            if action_key in used_actions:
                traj.n_repeated_actions += 1
            else:
                used_actions.append(action_key)

            # ---- 执行动作 ----
            action_str = f"{action_name}[{action_arg}]"
            if action_name == "search":
                traj.search_attempts += 1
            elif action_name == "lookup":
                traj.lookups += 1

            obs_obj = self.env.step(action_str)
            obs = obs_obj.text.replace("\\n", "")
            if action_name == "search" and not obs.startswith("Could not find"):
                traj.search_hits += 1

            history_lines.append(self._format_step(method, step, thought, action_str, obs))

            if action_name == "finish":
                traj.completed = True
                traj.prediction = action_arg
                break

        # 步数用尽：官方代码调用 finish[]（空答案），EM 记 0
        if not traj.completed:
            traj.prediction = ""
            traj.error = traj.error or "max_steps_exhausted"

        traj.history = "\n".join(history_lines)
        traj.metrics = compute_metrics(traj.prediction, gold)
        return traj

    # ------------------------------------------------------------------
    @staticmethod
    def _parse_first_step(out: str, method: str, step: int) -> Tuple[str, str, str]:
        """从模型输出中解析**第一段** (action_name, action_arg, thought)。

        模型可能一次生成多步轨迹（如 Thought1/Action1/Observation1/Thought2/...）。
        由于 Observation 必須由環境真實執行得到（不可由模型臆造），本方法只取
        第一段的 Thought 與 Action，其餘部分交回下一輪由環境反饋驅動。

        對應官方实现的切分規則：官方按 "\\nAction {i}: " 切分，這裡改為
        「切到第一個 Action 之前為 thought，第一個 Action 之後到下一個
        Observation/Thought 之前為 action」。
        """
        seg = (out or "").strip()
        if not seg:
            return "", "", ""

        # 切出第一段：遇到 Observation 或下一個 Thought 即停止
        # ReAct 格式：Thought i: ... \n Action i: ...
        # Act 格式：Action i: ...
        if method == "react":
            m = re.search(r"Thought\s*\d*\s*[:：]\s*(.*?)(?=\n\s*Action\s*\d*\s*[:：])",
                          seg, re.DOTALL | re.IGNORECASE)
            thought = m.group(1).strip() if m else ""
            # 動作部分：從第一個 Action 開始，到下一個 Observation 或 Thought 為止
            am = re.search(r"Action\s*\d*\s*[:：]\s*(.*?)(?=\n\s*(?:Observation\s*\d*\s*[:：]|Thought\s*\d*\s*[:：])|\Z)",
                           seg, re.DOTALL | re.IGNORECASE)
            action_text = am.group(1) if am else seg
        else:
            thought = ""
            am = re.search(r"Action\s*\d*\s*[:：]\s*(.*?)(?=\n\s*(?:Observation\s*\d*\s*[:：]|Action\s*\d*\s*[:：])|\Z)",
                           seg, re.DOTALL | re.IGNORECASE)
            action_text = am.group(1) if am else seg

        name, arg = parse_action(action_text)
        if not name:
            # 退回：直接對整段做寬鬆匹配
            name, arg = parse_action(seg)
        return name, arg, thought

    # ------------------------------------------------------------------
    def _retry_action(
        self, few_shot: str, question: str, history: str, thought: str,
        step: int, method: str, traj: Trajectory,
    ) -> Tuple[str, str]:
        """官方的 badcall 补救：补一次调用单独生成 Action。

        与主路径一致，prompt 不内联续写前缀（实测内联前缀会导致模型只复述前缀）。
        为让模型只产出动作，这里在 prompt 中明确要求「只输出 Action 行」。
        """
        for _ in range(self.max_generation_retries):
            traj.n_badcalls += 1
            hint = (
                f"\n(Your previous attempt failed to produce a valid action. "
                f"Output ONLY the next action line in the format "
                f"'Action {step}: Search[...]' or 'Action {step}: Lookup[...]' "
                f"or 'Action {step}: Finish[...]'.)"
            )
            base = self._build_step_prompt(few_shot, question, history, method, step)
            # 在最終的 user 內容後附加提示
            marker = "<|im_start|>user\n"
            idx = base.rfind(marker)
            if idx != -1:
                user_content = base[idx + len(marker):].rsplit("<|im_end|>", 1)[0]
                prompt = (base[:idx + len(marker)] + user_content + hint
                          + "<|im_end|>\n<|im_start|>assistant\n")
            else:
                prompt = base + hint
            try:
                out = self.llm.generate([prompt], stop_sequences=[[]])[0]
            except Exception:  # noqa: BLE001
                continue
            traj.raw_outputs.append(out)
            name, arg, _ = self._parse_first_step(out, method, step)
            if name:
                return name, arg
        return "", ""

    @staticmethod
    def _format_step(method: str, step: int, thought: str, action: str, obs: str) -> str:
        """按官方格式拼接轨迹片段。"""
        lines = []
        if method == "react" and thought:
            lines.append(f"Thought {step}: {thought}")
        if action:
            lines.append(f"Action {step}: {action}")
        if obs:
            lines.append(f"Observation {step}: {obs}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def _build_step_prompt(self, few_shot: str, question: str, history: str,
                           method: str, step: int) -> str:
        """构造第 step 步的 prompt。

        【实测得到的关键设计】
        1. 3B instruct 模型若在 prompt 末尾内联 'Thought {i}:' 之类续写前缀，会把
           前缀当成待复述内容，输出恰好等于前缀就停 —— 因此**不内联前缀**。
        2. 模型若看到完整的历史轨迹，会倾向于**从头重写整条轨迹**而不是续写下一步。
           解决办法是明确告知「Continue from where the previous attempt stopped」，
           并在 few-shot 之后、历史之前显式给出 'Continue the solution.' 指令。
        3. 历史中的 Observation 由环境真实执行得到，必须以明确的分隔标记呈现，
           让模型知道哪些是「已完成的事实」、哪些是「要继续的部分」。
        """
        instruction = (
            "Below are examples of the required output format. Follow the same "
            "format exactly.\n\n"
        )
        if history:
            instruction += (
                "Continue the solution from where it stops. Do NOT restart from the "
                "beginning. Output ONLY the next step (a 'Thought' line and/or an "
                "'Action' line).\n\n"
            )
        user_content = instruction + few_shot + "\n\n" + f"Question: {question}"
        if history:
            user_content += "\n" + history
            user_content += "\n(Continue with the next step.)"
        messages = [{"role": "user", "content": user_content}]
        return self.llm.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
