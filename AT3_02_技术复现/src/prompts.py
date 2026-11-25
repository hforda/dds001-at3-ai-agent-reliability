"""
prompts.py — ReAct 的四种 prompting 格式与 few-shot 示例（从零实现）

严格依据论文附录 C.1（HotpotQA）逐字转录，并遵循论文 §3.2 的消融构造原则：
  - Standard：移除 ReAct 轨迹中的所有 thought / action / observation
  - CoT   ：移除 action 与 observation，作为纯推理基线
  - Act   ：移除 thought（**与 ReAct 是同一批人工轨迹**，构成严格受控消融）
  - ReAct ：完整轨迹

【格式细节说明】
论文附录表格排版丢失了 'Action 1' / 'Observation 1' 的冒号，官方代码 prompts_naive.json
则带冒号。本次复现采用**官方代码格式**（带冒号），因为模型需要从 few-shot 中学到稳定的
输出格式，否则解析会失败。附录的 observation 内容与官方代码一致。

CoT 示例的 'Let's think step by step.' 前缀与结尾 'so the answer is X.' 均为论文附录原文。
"""
from __future__ import annotations

from typing import List, Tuple

# ---------------------------------------------------------------------------
# 6 条 HotpotQA few-shot 示例的 question / answer（论文附录 C.1）
# ---------------------------------------------------------------------------
QUESTIONS: List[str] = [
    "What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?",
    'Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?',
    "Which documentary is about Finnish rock groups, Adam Clayton Powell or The Saimaa Gesture?",
    "What profession does Nicholas Ray and Elia Kazan have in common?",
    "Which magazine was started first Arthur's Magazine or First for Women?",
    "Were Pavel Urysohn and Leonid Levin known for the same type of work?",
]

ANSWERS: List[str] = [
    "1,800 to 7,000 ft",
    "Richard Nixon",
    "The Saimaa Gesture",
    "director, screenwriter, actor",
    "Arthur's Magazine",
    "yes",
]

# ---------------------------------------------------------------------------
# Act / ReAct 轨迹：observation 文本取自论文附录 C.1（官方代码同源）
# ---------------------------------------------------------------------------
ACT_TRAJECTORIES: List[List[Tuple[str, str]]] = [
    [  # Q1
        ("Search[Colorado orogeny]",
         "The Colorado orogeny was an episode of mountain building (an orogeny) in Colorado and surrounding areas."),
        ("Lookup[eastern sector]",
         "(Result 1 / 1) The eastern sector extends into the High Plains and is called the Central Plains orogeny."),
        ("Search[High Plains]",
         "High Plains refers to one of two distinct land regions:"),
        ("Search[High Plains (United States)]",
         "The High Plains are a subregion of the Great Plains. From east to west, the High Plains rise in elevation from around 1,800 to 7,000 ft (550 to 2,130 m).[3]"),
        ("Finish[1,800 to 7,000 ft]", None),
    ],
    [  # Q2
        ("Search[Milhouse]",
         "Milhouse Mussolini Van Houten is a recurring character in the Fox animated television series The Simpsons voiced by Pamela Hayden and created by Matt Groening."),
        ("Lookup[named after]",
         "(Result 1 / 1) Milhouse was named after U.S. president Richard Nixon, whose middle name was Milhous."),
        ("Finish[Richard Nixon]", None),
    ],
    [  # Q3
        ("Search[Adam Clayton Powell]",
         "Could not find [Adam Clayton Powell]. Similar: ['Adam Clayton Powell III', 'Seventh Avenue (Manhattan)', 'Adam Clayton Powell Jr. State Office Building', 'Isabel Washington Powell', 'Adam Powell', 'Adam Clayton Powell (film)', 'Giancarlo Esposito']."),
        ("Search[Adam Clayton Powell (film)]",
         "Adam Clayton Powell is a 1989 American documentary film directed by Richard Kilberg. The film is about the rise and fall of influential African-American politician Adam Clayton Powell Jr.[3][4] It was later aired as part of the PBS series The American Experience."),
        ("Finish[The Saimaa Gesture]", None),
    ],
    [  # Q4
        ("Search[Nicholas Ray]",
         "Nicholas Ray (born Raymond Nicholas Kienzle Jr., August 7, 1911 – June 16, 1979) was an American film director, screenwriter, and actor best known for the 1955 film Rebel Without a Cause."),
        ("Search[Elia Kazan]",
         "Elia Kazan was an American film and theatre director, producer, screenwriter and actor."),
        ("Finish[director, screenwriter, actor]", None),
    ],
    [  # Q5
        ("Search[Arthur's Magazine]",
         "Arthur's Magazine (1844--1846) was an American literary periodical published in Philadelphia in the 19th century."),
        ("Search[First for Women]",
         "First for Women is a woman's magazine published by Bauer Media Group in the USA.[1] The magazine was started in 1989."),
        ("Finish[Arthur's Magazine]", None),
    ],
    [  # Q6
        ("Search[Pavel Urysohn]",
         "Pavel Samuilovich Urysohn (February 3, 1898 - August 17, 1924) was a Soviet mathematician who is best known for his contributions in dimension theory."),
        ("Search[Leonid Levin]",
         "Leonid Anatolievich Levin is a Soviet-American mathematician and computer scientist."),
        ("Finish[yes]", None),
    ],
]

# ReAct 轨迹的 thought 序列（论文附录 C.1 原文；末条为综合答案的 thought）
REACT_THOUGHTS: List[List[str]] = [
    [  # Q1
        "I need to search Colorado orogeny, find the area that the eastern sector of the Colorado orogeny extends into, then find the elevation range of the area.",
        "It does not mention the eastern sector. So I need to look up eastern sector.",
        "The eastern sector of Colorado orogeny extends into the High Plains. So I need to search High Plains and find its elevation range.",
        "I need to instead search High Plains (United States).",
        "High Plains rise in elevation from around 1,800 to 7,000 ft, so the answer is 1,800 to 7,000 ft.",
    ],
    [  # Q2
        'The question simplifies to "The Simpsons" character Milhouse is named after who. I only need to search Milhouse and find who it is named after.',
        'The paragraph does not tell who Milhouse is named after, maybe I can look up "named after".',
        "Milhouse was named after U.S. president Richard Nixon, so the answer is Richard Nixon.",
    ],
    [  # Q3
        "I need to search Adam Clayton Powell and The Saimaa Gesture, and find which documentary is about Finnish rock groups.",
        "To find the documentary, I can search Adam Clayton Powell (film).",
        "Adam Clayton Powell (film) is a documentary about an African-American politician, not Finnish rock groups. So the documentary about Finnish rock groups must instead be The Saimaa Gesture.",
    ],
    [  # Q4
        "I need to search Nicholas Ray and Elia Kazan, find their professions, then find the profession they have in common.",
        "Professions of Nicholas Ray are director, screenwriter, and actor. I need to search Elia Kazan next and find his professions.",
        "Professions of Elia Kazan are director, producer, screenwriter, and actor. So profession Nicholas Ray and Elia Kazan have in common is director, screenwriter, and actor.",
    ],
    [  # Q5
        "I need to search Arthur's Magazine and First for Women, and find which was started first.",
        "Arthur's Magazine was started in 1844. I need to search First for Women next.",
        "First for Women was started in 1989. 1844 (Arthur's Magazine) < 1989 (First for Women), so Arthur's Magazine was started first.",
    ],
    [  # Q6
        "I need to search Pavel Urysohn and Leonid Levin, find their types of work, then find if they are the same.",
        "Pavel Urysohn is a mathematician. I need to search Leonid Levin next and find its type of work.",
        "Pavel Urysohn is a mathematician and computer scientist. So Pavel Urysohn and Leonid Levin have the same type of work.",
    ],
]

# CoT 的 thought 序列（论文附录 C.1 原文）
COT_THOUGHTS: List[str] = [
    "Let's think step by step. The eastern sector of Colorado orogeny extends into the High Plains. High Plains rise in elevation from around 1,800 to 7,000 ft, so the answer is 1,800 to 7,000 ft.",
    "Let's think step by step. Milhouse was named after U.S. president Richard Nixon, so the answer is Richard Nixon.",
    "Let's think step by step. Adam Clayton Powell (film) is a documentary about an African-American politician, not Finnish rock groups. So the documentary about Finnish rock groups must instead be The Saimaa Gesture.",
    "Let's think step by step. Professions of Nicholas Ray are director, screenwriter, and actor. Professions of Elia Kazan are director, producer, screenwriter, and actor. So profession Nicholas Ray and Elia Kazan have in common is director, screenwriter, and actor.",
    "Let's think step by step. Arthur's Magazine was started in 1844. First for Women was started in 1989. 1844 (Arthur's Magazine) < 1989 (First for Women), so Arthur's Magazine was started first.",
    "Let's think step by step. Pavel Urysohn is a mathematician. Leonid Levin is a mathematician and computer scientist. So Pavel Urysohn and Leonid Levin have the same type of work.",
]


# ---------------------------------------------------------------------------
# 四种格式的 prompt 构造
# ---------------------------------------------------------------------------
def _header(method: str) -> str:
    """各格式在 few-shot 之前附加的简短说明。"""
    if method == "standard":
        return ""
    if method == "cot":
        return ""
    if method == "act":
        return ""
    if method == "react":
        return ""
    raise ValueError(f"unknown method: {method}")


def build_few_shot(method: str) -> str:
    """构造 6-shot 的示例块（不含待答问题）。"""
    if method == "standard":
        parts = []
        for q, a in zip(QUESTIONS, ANSWERS):
            parts.append(f"Question: {q}\nAnswer: {a}")
        return "\n\n".join(parts)

    if method == "cot":
        parts = []
        for q, th, a in zip(QUESTIONS, COT_THOUGHTS, ANSWERS):
            parts.append(f"Question: {q}\nThought: {th}\nAnswer: {a}")
        return "\n\n".join(parts)

    if method == "act":
        parts = []
        for q, traj in zip(QUESTIONS, ACT_TRAJECTORIES):
            lines = [f"Question: {q}"]
            for i, (act, obs) in enumerate(traj, start=1):
                lines.append(f"Action {i}: {act}")
                if obs is not None:
                    lines.append(f"Observation {i}: {obs}")
            parts.append("\n".join(lines))
        return "\n\n".join(parts)

    if method == "react":
        parts = []
        for q, traj, thoughts in zip(QUESTIONS, ACT_TRAJECTORIES, REACT_THOUGHTS):
            lines = [f"Question: {q}"]
            for i, (act, obs) in enumerate(traj, start=1):
                lines.append(f"Thought {i}: {thoughts[i - 1]}")
                lines.append(f"Action {i}: {act}")
                if obs is not None:
                    lines.append(f"Observation {i}: {obs}")
            parts.append("\n".join(lines))
        return "\n\n".join(parts)

    raise ValueError(f"unknown method: {method}")


def build_prompt(method: str, question: str, history: str = "") -> str:
    """构造完整 prompt。

    history 是已完成的轨迹片段（标准/CoT 为空；Act/ReAct 为已累积的
    'Thought i / Action i / Observation i' 文本）。
    """
    blocks = []
    head = _header(method)
    if head:
        blocks.append(head)
    blocks.append(build_few_shot(method))
    q_block = f"Question: {question}\n"
    if history:
        q_block += history
    blocks.append(q_block.rstrip("\n"))
    return "\n\n".join(blocks)


def method_requires_env(method: str) -> bool:
    """该方法是否需要 Wikipedia 环境交互。"""
    return method in ("act", "react")


def method_requires_thought(method: str) -> bool:
    """该方法是否生成 thought。"""
    return method in ("cot", "react")
