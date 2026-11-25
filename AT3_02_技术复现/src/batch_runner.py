"""
batch_runner.py — 跨题同步推进的批量执行器

【为什么需要】
逐题串行执行时，每一步的 GPU 生成都要等单题完成（实测 ~9s/步），
100 题 × 4 baseline × 最多 7 步的串行耗时不可接受。

本模块把所有题目放在同一个「步进循环」里：每一轮对**所有未结束的题目**同时
构造 prompt 并批量生成，然后逐题执行动作、累积各自的轨迹。

【关键正确性约束：每题必须有独立的 Wikipedia 环境实例】
WikipediaEnv 持有逐回合状态（current_page_text / current_page_title /
lookup_cache / lookup_cursor）。若多题共用一个实例，B 题的 `lookup` 会作用在
A 题刚检索到的页面上 —— 这是会静默污染结果的真实 bug。
因此本模块为每道题创建独立的 env 实例（共享磁盘缓存，因此不额外增加网络开销），
并利用这一点在同一轮内**并行**执行各题的环境调用（Wikipedia API 是 I/O 密集）。

【与官方实现的等价性】
官方实现是逐题独立循环；本实现只是把「同一轮次、不同题目」的生成合并为一次
batch，**每道题的环境交互序列、轨迹内容、步数上限、终止条件完全不变**，
因此结果与逐题串行等价。
"""
from __future__ import annotations

import concurrent.futures
import logging
from typing import Dict, List, Optional, Tuple

from .agent import ReActAgent, Trajectory, extract_answer_line
from .evaluation import compute_metrics
from .llm import LocalLLM
from .prompts import build_few_shot
from .wikipedia_env import WikipediaEnv

LOGGER = logging.getLogger(__name__)


class _TimeoutObservation:
    """环境调用超时的替身（保持与 Observation 的鸭子类型兼容）。"""
    text = "Environment request timed out."
    done = False
    reward = 0.0
    info: Dict[str, str] = {}


class BatchReActRunner:
    """把 N 道题放在同一循环中推进，支持 standard / cot / act / react。"""

    def __init__(
        self,
        llm: LocalLLM,
        env: WikipediaEnv,
        max_steps: int = 7,
        max_consecutive_invalid: int = 2,
        max_generation_retries: int = 2,
        batch_size: int = 8,
        env_workers: int = 8,
        env_timeout: float = 120.0,
    ) -> None:
        self.llm = llm
        self.env_template = env
        self.max_steps = max_steps
        self.max_consecutive_invalid = max_consecutive_invalid
        self.max_generation_retries = max_generation_retries
        self.batch_size = batch_size
        self.env_workers = env_workers
        self.env_timeout = env_timeout
        self.agent = ReActAgent(
            llm=llm, env=env, max_steps=max_steps,
            max_consecutive_invalid=max_consecutive_invalid,
            max_generation_retries=max_generation_retries,
        )

    # ------------------------------------------------------------------
    def _new_env(self) -> WikipediaEnv:
        """为一道题创建独立的环境实例（共享磁盘缓存）。"""
        tpl = self.env_template
        env = WikipediaEnv(
            cache_dir=tpl.cache_dir,
            timeout=tpl.timeout,
            max_retries=tpl.max_retries,
            user_agent=tpl.session.headers.get("User-Agent", "AT3-ReAct-Reproduction/1.0"),
            search_sentences=tpl.search_sentences,
        )
        return env

    # ------------------------------------------------------------------
    def run(self, method: str, records: List[Dict[str, str]]) -> List[Trajectory]:
        """对全部题目执行指定方法。"""
        if method in ("standard", "cot"):
            return self._run_direct_batch(method, records)
        return self._run_interactive_batch(method, records)

    # ------------------------------------------------------------------
    def _run_direct_batch(self, method: str, records: List[Dict[str, str]]) -> List[Trajectory]:
        """Standard / CoT：一次性批量生成。"""
        trajs: List[Trajectory] = []
        few_shot = build_few_shot(method)
        prompts = [
            self.agent._build_step_prompt(few_shot, r["question"], "", method, 1)
            for r in records
        ]
        outs = self.llm.generate(prompts, stop_sequences=[[] for _ in prompts])
        for r, out in zip(records, outs):
            traj = Trajectory(qid=r["qid"], question=r["question"],
                              gold_answer=r["gold_answer"], method=method)
            traj.raw_outputs.append(out)
            traj.steps = 1
            traj.completed = True
            traj.format_ok = bool(out.strip())
            traj.prediction = extract_answer_line(out)
            traj.metrics = compute_metrics(traj.prediction, r["gold_answer"])
            trajs.append(traj)
        return trajs

    # ------------------------------------------------------------------
    def _run_interactive_batch(self, method: str,
                               records: List[Dict[str, str]]) -> List[Trajectory]:
        """Act / ReAct：所有题目同步推进同一轮，各题使用独立环境实例。"""
        few_shot = build_few_shot(method)
        n = len(records)
        trajs: List[Trajectory] = [
            Trajectory(qid=r["qid"], question=r["question"],
                       gold_answer=r["gold_answer"], method=method)
            for r in records
        ]
        envs: List[WikipediaEnv] = [self._new_env() for _ in range(n)]
        histories: List[List[str]] = [[] for _ in range(n)]
        consecutive_invalid: List[int] = [0] * n
        used_actions: List[List[str]] = [[] for _ in range(n)]
        active: List[bool] = [True] * n

        for step in range(1, self.max_steps + 1):
            idxs = [i for i in range(n) if active[i]]
            if not idxs:
                break
            LOGGER.info("  step %d/%d：%d 题待处理", step, self.max_steps, len(idxs))

            # ---- 构造 prompt 并批量生成 ----
            prompts = [
                self.agent._build_step_prompt(
                    few_shot, records[i]["question"], "\n".join(histories[i]), method, step
                )
                for i in idxs
            ]
            outs = self.llm.generate(prompts, stop_sequences=[[] for _ in prompts])

            # ---- 解析动作（逐题）----
            actions: Dict[int, Tuple[str, str, str]] = {}
            for i, out in zip(idxs, outs):
                traj = trajs[i]
                traj.raw_outputs.append(out)
                traj.steps = step
                name, arg, thought = self.agent._parse_first_step(out, method, step)
                if not name:
                    name, arg = self.agent._retry_action(
                        few_shot, records[i]["question"], "\n".join(histories[i]),
                        thought, step, method, traj
                    )
                actions[i] = (name, arg, thought)

            # ---- 并行执行环境调用（各题环境独立，故可安全并行）----
            # 用整体超时保护：若个别请求挂起，不让整个实验被阻塞。
            valid = [i for i in idxs if actions[i][0]]
            if valid:
                import time as _time
                t_env = _time.time()
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=min(self.env_workers, len(valid))
                ) as ex:
                    futs = {
                        i: ex.submit(envs[i].step, f"{actions[i][0]}[{actions[i][1]}]")
                        for i in valid
                    }
                    obs_map = {}
                    for i in valid:
                        try:
                            obs_map[i] = futs[i].result(timeout=self.env_timeout)
                        except Exception as exc:  # noqa: BLE001
                            LOGGER.warning("  环境调用超时/失败 qid=%s: %s", i, exc)
                            obs_map[i] = _TimeoutObservation()
                LOGGER.info("    环境调用耗时 %.1fs", _time.time() - t_env)

            # ---- 汇总本轮结果 ----
            for i in idxs:
                traj = trajs[i]
                name, arg, thought = actions[i]
                if not name:
                    traj.n_invalid_actions += 1
                    consecutive_invalid[i] += 1
                    obs_text = f"Invalid action: {traj.raw_outputs[-1].strip()[:120]}"
                    histories[i].append(
                        self.agent._format_step(method, step, thought, "", obs_text)
                    )
                    if consecutive_invalid[i] >= self.max_consecutive_invalid:
                        active[i] = False
                    continue

                consecutive_invalid[i] = 0
                traj.n_actions += 1
                traj.format_ok = True

                action_key = f"{name}[{arg.lower()}]"
                if action_key in used_actions[i]:
                    traj.n_repeated_actions += 1
                else:
                    used_actions[i].append(action_key)

                if name == "search":
                    traj.search_attempts += 1
                elif name == "lookup":
                    traj.lookups += 1

                action_str = f"{name}[{arg}]"
                obs_text = obs_map[i].text.replace("\\n", "")
                if name == "search" and not obs_text.startswith("Could not find"):
                    traj.search_hits += 1

                histories[i].append(
                    self.agent._format_step(method, step, thought, action_str, obs_text)
                )

                if name == "finish":
                    traj.completed = True
                    traj.prediction = arg
                    active[i] = False

        # 收尾：未完成的题目按官方规则强制 finish[]（空答案）
        for i, traj in enumerate(trajs):
            traj.history = "\n".join(histories[i])
            if not traj.completed:
                traj.prediction = ""
                traj.error = traj.error or "max_steps_exhausted"
            traj.metrics = compute_metrics(traj.prediction, traj.gold_answer)
        return trajs
