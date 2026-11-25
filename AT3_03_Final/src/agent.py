"""
agent.py — AT3 ③ 的 Agent 工具調用循環

【關鍵設計：Agent 是真的由模型決策，不是預設順序】
每一步的流程：
  1. 把「任務 + 工具說明 + 已完成步驟 + 最近的 observation」組成 prompt
  2. **由本地模型生成下一步的動作**（Action: search_sources(query=...) 等）
  3. 解析模型的動作，執行工具，記錄 observation
  4. 把 observation 餵回 context，進入下一步

模型輸出格式（引導自 ReAct 風格，但改為本任務的三工具）：
    Thought: <推理>
    Action: <tool_name>(<param>=<value>, ...)
    （若要結束）Action: write_brief(brief_text="...", cited_doc_ids=[...])

模型若輸出無法解析的動作 → 記錄為 format_error 並把錯誤訊息回饋給模型重試
（這是 Agent 的真實失敗模式，會被記錄到 error-note.md）。

停止條件（來自 task-contract）：
  - 模型呼叫 write_brief → 任務完成
  - step 數達到 max_steps → 安全停止（未完成）
  - 模型連續 K 次無法解析動作 → 安全停止

【模型】
Qwen2.5-Coder-3B-Instruct（AT3 ② 已下載的本地權重，路徑透過 REACT_MODEL_PATH 指定）
不訓練、不微調、不呼叫 API。
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tools import SourcePack, ToolCall, ToolBox  # noqa: E402

# ---------------------------------------------------------------------------
# Prompt 模板（兩組實驗逐字相同，唯一差異是 max_steps 參數）
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a careful research assistant working OFFLINE on a local document source pack.

You must complete this task by calling tools, one step at a time:
{task}

Available tools:
{tool_list}

Rules:
1. You may ONLY use the three tools listed above.
2. Use exactly this action format on a single line:
   Action: tool_name(parameter=value, parameter2=value2)
   The angle-bracket values below are PLACEHOLDERS, never real ids:
   Action: search_sources(query="large language model evaluation")
   Action: read_source(doc_id="<doc_id_from_search_results>")
   Action: write_brief(brief_text="<the markdown brief>", cited_doc_ids=["<doc_id_1>","<doc_id_2>","<doc_id_3>"])
   Replace every placeholder with a real value that came from your own tool output:
   - the query in search_sources is text you write yourself;
   - the doc_id in read_source MUST be one of the doc_ids that the most recent
     search_sources call returned. Reading any other id is REJECTED by the tool.
   - the doc_ids in write_brief MUST be ids you have already read.
   Never invent, guess or recall a doc_id from memory. If you did not see it in a
   search_sources result, you may not use it.
   Every doc_id has the form NNNN.NNNNNvN (four digits, a dot, five digits, then vN).
   If a doc_id you are about to use does not have that shape, it is wrong - do not
   use it. Copy doc_ids character by character from the search output above.
3. REQUIRED ORDER - you MUST follow it:
   (a) Call search_sources ONCE with a good query. One search is normally enough.
   (b) Then call read_source on THREE DIFFERENT doc_id values (three separate calls),
       each copied exactly from the candidates returned by search_sources.
       Choose the three whose titles and matching lines are most about how LLMs are
       evaluated. Do not read the same document twice.
   (c) Only after you have read three documents, call write_brief.
   If you call write_brief before reading three documents it will be REJECTED and you will
   lose a step. Keep track of which doc_ids you have already read.
   IMPORTANT: put cited_doc_ids LAST in the argument list.
   STEP BUDGET: you have a limited number of steps. The shortest path is
       search_sources -> read_source -> read_source -> read_source -> write_brief.
   Do not search again, do not re-read a document you already read, and do not write
   the brief until you have three different documents in hand.
4. Search results do NOT contain full text. You MUST call read_source before citing.
5. HOW TO WRITE A CITED BULLET - this is the whole point of the task:
   A read_source result shows numbered lines, each beginning with a label like s[7].
   Pick ONE of those lines and build your bullet by COPYING that entire line's text
   and appending the marker, in this shape:
       - <the text of the line you copied, word for word> [doc_id:<that doc's id>#s<that line's number>]
   Rules for the copy - READ CAREFULLY, this is where briefs get rejected:
   - The bullet text must be IDENTICAL to the line you copied. The validator compares
     the two character by character. If you change even one word - including
     replacing "In this paper, we propose" with the authors' names, or dropping
     "we", or swapping "First,"/"Furthermore,"/"Finally," - the brief is REJECTED.
   - The bullet text must NOT be the document title and must NOT be a phrase you
     wrote yourself. It must be the literal text of one s[] line.
   - Copy the WHOLE line, from its first word to its last, including its opening words
     and any trailing period.
   - Do not shorten it, do not paraphrase it, do not fix its grammar, do not drop the
     last few words, do not retype it from memory.
   - The number after #s must be the same number as the s[] label of the line you copied.
   - If a line is too long to copy entirely, pick a shorter line instead - never
     truncate a line.
   - The validator checks the copied text appears character for character in that bullet.
     If the text is not there, the whole brief is rejected and you lose a step.
   - NEVER invent the text of a bullet. Every bullet's text must come from a line you
     actually saw in a read_source result. If you cannot remember the exact wording,
     re-read the document rather than guessing.
   - NEVER write your own summary of a paper as a bullet. A bullet is ALWAYS one of the
     s[] lines shown above, copied word for word. Writing a summary sentence such as
     "Bayesian lens, in-context learning, and latent variable models are used to
     evaluate LLMs" is REJECTED, because that wording appears in no source line.
   - Pick the line that says the most about the evaluation method, not necessarily the
     first one.
   - You need at least one bullet per document you read, so at least three bullets,
     and the three bullets must come from three different documents.
   Put the most informative lines you found as your bullets. Do not use the first line
   of a document if a later line says more about the evaluation method.
   - Every document you read must contribute at least one bullet. Check before you
     write: one bullet from document A, one from document B, one from document C.
   - A bullet must NOT be the paper's title, and must NOT be your own summary of the
     paper. It must be one of the s[] lines you were shown. For example, if the lines
     you saw for a document were "s[4] The evaluation of the quality of the generated
     text is a complex problem." then the bullet for that document is that sentence,
     with its marker - not the paper's title.
   - Concretely: a bullet that reads like a paper title (capitalised, no ending
     period, no verbs) is WRONG. A bullet that reads like a sentence from the paper
     (starts with a capital letter, contains verbs, ends with a period) is RIGHT.
   - Keep the bullet's text EXACTLY as the s[] line, including its capitalisation and
     its ending period. Do not re-capitalise or re-punctuate it.

   DOC_IDS YOU HAVE READ (use these exact strings):
{cite_hints}

6. WRITE THE BRIEF like this (copy this shape, replace the placeholders):
   # <short title>

   <one or two sentences saying what the three documents are about>

   - <text copied from a line you read> [doc_id:<that id>#s<that number>]
   - <text copied from a line you read> [doc_id:<that id>#s<that number>]
   - <text copied from a line you read> [doc_id:<that id>#s<that number>]

   IMPORTANT: the bullet text must be the s[] line COPIED WORD FOR WORD, with its
   original capitalisation and its ending period. Do not re-write it in your own
   words, do not capitalise it differently, and do not add or remove words.
   If you are copying "s[4] The evaluation of the quality of the generated text is a
   complex problem." then your bullet is exactly:
       - The evaluation of the quality of the generated text is a complex problem. [doc_id:<id>#s4]
   Copy it. Do not summarise it. Do not change its capitalisation.
   - Do NOT put a label before the sentence. A bullet is only:
       - <the copied sentence> [doc_id:<id>#s<N>]
   Do NOT write "- **Evaluation Object**: <sentence>" or any similar prefix -
   the bullet must start with the copied sentence itself.
   - Do NOT use bold/italic markers (**, *) anywhere in the bullet text.

   - Each document block below shows exactly ONE s[] line that you may cite.
     Copy that line word for word into the bullet for that document.
   - Exactly three bullets: bullet 1 from document 1, bullet 2 from document 2,
     bullet 3 from document 3 - one from each, never two from the same document.
   - Every bullet ends with a marker, and each of the three doc_ids appears at least once.
   - Keep the summary sentences SHORT and in your own words; only the bullets must be
     copied word for word.
   - Plain Markdown only: no code blocks (```), no repeated lines, and nothing after
     the closing quote of brief_text.
7. Think briefly in the "Thought:" line, then give exactly one "Action:" line.
"""

ACTION_RE = re.compile(
    r"Action\s*:\s*(search_sources|read_source|write_brief)\s*\(",
    re.IGNORECASE,
)
# 平衡括號掃描：從 tool_name( 開始，取到配對的右括號
_BAL = {"(": ")", "[": "]", "{": "}"}


def _scan_args(text: str, open_idx: int) -> Optional[str]:
    """從 text[open_idx]（指向 '('）掃描配對括號內容，支援巢狀與字串內括號。"""
    opener = text[open_idx]
    closer = _BAL[opener]
    depth = 0
    in_str: Optional[str] = None
    i = open_idx
    while i < len(text):
        ch = text[i]
        if in_str:
            if ch == "\\":
                i += 2
                continue
            if ch == in_str:
                in_str = None
        elif ch in ('"', "'"):
            in_str = ch
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[open_idx + 1:i]
        i += 1
    return None


KV_RE = re.compile(
    r"""([A-Za-z_][A-Za-z0-9_]*)\s*=\s*("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|\[[^\]]*\]|[^,]*)""",
    re.DOTALL,
)

def _strip_repetition(text: str) -> str:
    """移除模型退化產生的無限重複片段（如連續的 ``` ``` ```）。

    實測：3B 模型在長輸出時會退化為重複的 markdown 程式碼圍欄，
    這會破壞 brief_text 的閉引號，導致參數解析失敗。
    這是真實的模型行為，此處做防禦性清理並在 error-note 中記錄。
    """
    if not text:
        return text
    # 連續 3 次以上的相同行 → 只保留第一次
    lines = text.split("\n")
    out: List[str] = []
    prev = None
    run = 0
    for ln in lines:
        key = ln.strip()
        if key and key == prev:
            run += 1
            if run >= 2:
                continue
        else:
            run = 0
        out.append(ln)
        prev = key if key else prev
    return "\n".join(out)


_CONTENT_STOP = {
    "that", "this", "with", "from", "have", "been", "were", "they", "which",
    "their", "there", "these", "those", "about", "would", "could", "should",
    "using", "used", "such", "also", "than", "then", "when", "what", "into",
    "over", "more", "most", "some", "each", "both", "does", "make", "made",
}


def _content_words_of(s: str) -> set:
    """取內容詞（去掉停用詞），用於句級語意對齊。"""
    return {w for w in re.findall(r"[A-Za-z][A-Za-z\-']{3,}", s.lower())
            if w not in _CONTENT_STOP}


def _parse_kv(raw_args: str) -> Dict[str, Any]:
    """把工具呼叫的參數串解析成 dict。支援雙引號、單引號、陣列與裸值。"""
    args: Dict[str, Any] = {}
    for km in KV_RE.finditer(raw_args):
        key = km.group(1)
        val = km.group(2).strip()
        if val.startswith('"') and val.endswith('"'):
            val = val[1:-1]
            # 模型有時輸出 \\n（雙反斜線）而非 \n；兩者都還原為換行
            val = val.replace("\\n", "\n").replace("\\t", "\t")
            val = val.replace('\\"', '"').replace("\\\\", "\\")
        elif val.startswith("'") and val.endswith("'"):
            # 單引號字串：模型常寫 doc_id='2301.12004v1'
            val = val[1:-1].replace("\\n", "\n").replace("\\'", "'")
        elif val.startswith("["):
            inner = val[1:-1].strip()
            val = [x.strip().strip('"').strip("'") for x in inner.split(",") if x.strip()]
        args[key] = val
    return args


def parse_action(text: str) -> Optional[Dict[str, Any]]:
    """從模型輸出解析動作。失敗回傳 None（觸發 format_error 重試）。"""
    if not text:
        return None
    text = _strip_repetition(text)
    # 找出所有 "Action: tool_name(" 出現位置。
    # 實測：3B 模型有時會幻想出多輪對話（Assistant:/Human: 標籤），
    # 其中包含多個 Action。**模型自己的決策是第一個 Action**，
    # 後續的是幻覺內容，取第一個可完整解析者。
    candidates = list(re.finditer(
        r"Action\s*:\s*(search_sources|read_source|write_brief)\s*\(",
        text, re.IGNORECASE))
    if not candidates:
        return None
    best: Optional[Tuple[str, int, int]] = None
    for m in candidates:
        name = m.group(1).lower()
        open_idx = m.end() - 1
        raw_args = _scan_args(text, open_idx)
        if raw_args is None:
            continue
        args = _parse_kv(raw_args)
        # 第一個能解析出至少一個參數的動作即為模型決策
        if args or name == "write_brief":
            best = (name, m.start(), open_idx)
            break
    if best is None:
        return None
    name, _, open_idx = best
    raw_args = _scan_args(text, open_idx)
    if raw_args is None:
        raw_args = ""
    args = _parse_kv(raw_args)
    # 退化輸出可能讓閉引號缺失 → 補救：若 brief_text 為空但原始輸出含該鍵，
    # 從原始輸出截取到最後一個未閉合引號為止的內容。
    # 實測：3B 模型在輸出接近 token 上限時會截斷，導致閉引號遺失、參數全丟。
    if name == "write_brief" and not args.get("brief_text"):
        m = re.search(r"brief_text\s*=\s*\"(.*)$", text, re.DOTALL)
        if m:
            body = m.group(1)
            # 去掉尾隨的重複圍欄與未閉合引號殘留
            body = re.sub(r"[\s`]+$", "", body)
            if body.strip():
                args["brief_text"] = body.strip()
    # 另一種常見形態：模型把簡報放進 markdown 程式碼圍欄，參數寫成 ...。
    # 實測（3B 模型）：Action: write_brief(...) ```Markdown ... ```
    # 此時 brief 的真實內容在圍欄內，必須取出，否則 brief_text 會是空的。
    if name == "write_brief" and not args.get("brief_text"):
        fences = re.findall(r"```(?:[A-Za-z]*)\s*\n(.*?)```", text, re.DOTALL)
        if fences:
            cand = max(fences, key=len).strip()
            if cand.startswith("#"):
                args["brief_text"] = cand
        else:
            # 圍欄未閉合（被截斷）→ 取最後一個圍欄之後的全部內容
            idx = text.rfind("```")
            if idx != -1:
                tail = text[idx:].split("\n", 1)
                if len(tail) > 1:
                    cand = tail[1].strip()
                    if cand.startswith("#"):
                        args["brief_text"] = cand
    return {"tool": name, "arguments": args}


class AgentRunner:
    """以本地模型驅動的工具調用循環。"""

    def __init__(
        self,
        llm,
        toolbox: ToolBox,
        task: str,
        max_steps: int = 6,
        max_format_errors: int = 2,
    ) -> None:
        self.llm = llm
        self.tb = toolbox
        self.task = task
        self.max_steps = max_steps
        self.max_format_errors = max_format_errors
        self.transcript: List[str] = []      # 給模型看的對話歷史（文字）
        self.steps_log: List[Dict[str, Any]] = []  # 結構化逐步紀錄（供 trajectory）
        self._current_step = 0
        self.alignment_warnings: List[str] = []    # 引用核驗診斷（不阻斷，僅記錄）
        self.brief_write_input: Optional[str] = None   # 寫入的原始輸入（無後處理）
        self.brief_write_sha256: Optional[str] = None  # 寫入內容的 SHA256
        self.format_errors = 0

    # ------------------------------------------------------------------
    def _tool_list_text(self) -> str:
        from contract import TOOL_SCHEMA
        out = []
        for t in TOOL_SCHEMA["tools"]:
            params = ", ".join(
                f"{p}: {v.get('type','string')}"
                for p, v in t["parameters"]["properties"].items()
            )
            out.append(f"  - {t['name']}({params}) — {t['description']}")
        return "\n".join(out)

    def _build_prompt(self) -> str:
        # 已讀 doc_id 清單：直接注入 prompt。實測 3B 模型無法自行構造
        # 正確的 arXiv doc_id 字串，必須把字面量給它照抄。這是 prompt 工程，
        # 不是把決策交給程式——選哪些 doc_id 仍由模型決定。
        read_ids = self.tb.called_docs()
        if read_ids:
            # 標註哪些已讀來源尚未被引用——這是最常見的失敗原因
            # （模型讀了三篇卻只引用了兩篇，導致 validator 的三來源檢查失敗）。
            used = set()
            for blk in self.transcript:
                if blk.startswith("Thought:") and "write_brief" in blk:
                    used = set(re.findall(r"\[doc_id:([^\]#\s]+)#s\d+\]", blk))
            hints = []
            for d in read_ids:
                flag = "  (already cited)" if d in used else "  <-- you MUST cite this one"
                hints.append(f"   {d}{flag}")
            hints = "\n".join(hints)
        else:
            hints = "   (none yet - call search_sources and read_source first)"
        sys_msg = SYSTEM_PROMPT.format(task=self.task, tool_list=self._tool_list_text(),
                                       cite_hints=hints)
        # 控制 context 長度：已讀文檔的 observation 不重複餵回全文，
        # 但**必須保留句子清單**——那是模型撰寫 citation 的唯一依據。
        # 做法：對每份已讀文檔，保留其 s[] 行（這是不可壓縮的證據）；
        # 其他步驟的觀察（錯誤訊息等）只保留最近兩步。
        keep_full = 2
        history_parts: List[str] = []
        n = len(self.transcript)
        for i, blk in enumerate(self.transcript):
            if i < n - keep_full and blk.startswith("Observation: READ"):
                # 壓縮 read_source 的 observation：保留 s[] 行，去掉表頭
                s_lines = [ln for ln in blk.split("\n")
                           if ln.startswith("s[") or ln.startswith("Action:")
                           or ln.startswith("Thought:")]
                history_parts.append("\n".join(s_lines))
            elif i < n - keep_full:
                action_lines = [ln for ln in blk.split("\n") if ln.startswith("Action:")]
                history_parts.append(action_lines[0] if action_lines else "(earlier step)")
            else:
                history_parts.append(blk)
        history = "\n\n".join(history_parts)
        return f"{sys_msg}\n\n--- CONVERSATION ---\n{history}\n\nYour next step:\nThought:"
    # ------------------------------------------------------------------
    def _build_messages(self) -> List[Dict[str, str]]:
        """以 chat template 所需的訊息列表構造本次生成的輸入。

        結構：system（規則與任務）+ user（目前狀態與已讀證據）。
        不把完整對話歷史餵回去——實測 3B 模型在長對話中會模仿歷史的
        "Action:/Observation:" 格式而無法產出新的規劃；改為每步重述
        當前狀態（已讀來源清單 + 各來源的 s[] 句子），既保留全部證據，
        又避免模型陷入續寫對話。
        """
        sys_msg = SYSTEM_PROMPT.format(task=self.task,
                                       tool_list=self._tool_list_text(),
                                       cite_hints=self._cite_hints_text())
        state: List[str] = ["CURRENT STATE:"]
        read_ids = self.tb.called_docs()
        n_searched = sum(1 for t in self.tb.calls if t.tool == "search_sources")
        if n_searched:
            state.append(f"You already ran search_sources {n_searched} time(s). "
                         f"Do NOT search again - use the candidate list below.")
        else:
            state.append("You have not searched yet. Your first action must be "
                         "search_sources.")
        if read_ids:
            state.append(f"You have read {len(read_ids)} document(s): {read_ids}")
        else:
            state.append("You have not read any document yet.")
        pool = self.tb.all_candidates or self.tb.last_candidates
        if pool:
            unread = [c for c in pool if c not in read_ids]
            state.append(f"doc_ids returned by your search_sources call(s): "
                         f"{pool}")
            if unread:
                state.append(f"NOT YET READ, pick one of these next: {unread}")
                state.append(
                    f"YOUR NEXT ACTION: call read_source on ONE of these doc_ids, "
                    f"for example read_source(doc_id=\"{unread[0]}\"). "
                    f"Do NOT call search_sources again - the candidates are already "
                    f"listed above.")
            else:
                state.append("All candidates have been read. Your next action must be "
                             "write_brief.")
        if len(read_ids) >= 3:
            mapping = "\n".join(
                f"   bullet {i} <- DOCUMENT BLOCK {i} = {d}"
                f"   (you MUST take one s[] line from THIS document)"
                for i, d in enumerate(read_ids, 1))
            state.append(
                f"YOU HAVE READ {len(read_ids)} DOCUMENTS - that is already enough. "
                f"You do NOT need to read any more. "
                f"YOUR NEXT ACTION MUST BE write_brief.\n"
                f"Write EXACTLY {len(read_ids)} bullets, one per document, in this "
                f"exact mapping (three bullets from ONE document is REJECTED):\n"
                f"{mapping}\n"
                f"For each bullet: copy ONE s[] line word for word from that "
                f"document's block, then append [doc_id:<that id>#s<that line number>].")
        else:
            need = 3 - len(read_ids)
            state.append(
                f"You still need to read {need} more document(s). "
                f"YOUR NEXT ACTION MUST BE read_source on one of the not-yet-read "
                f"doc_ids listed above. Do NOT call search_sources.")
        # 只呈現 s[] 句子，**不呈現標題**。
        # 實測：3B 模型看到標題時會直接把標題當成 bullet 的文字
        # （"**The effectiveness of very large language models on dialog evaluation**"），
        # 導致逐字核驗失敗。移除標題後，模型只能從 s[] 句子中選取 bullet 文字。
        for did in read_ids:
            doc = self.tb.pack.by_id.get(did)
            if not doc:
                continue
            doc_idx = read_ids.index(did) + 1
            lines = [f"\n=== DOCUMENT BLOCK {doc_idx}: {did} ===",
                     f"(paper title for your reference only - DO NOT use it as a "
                     f"bullet: {doc['title'][:100]})",
                     f"(take EXACTLY ONE bullet from the s[] lines below; its marker "
                     f"must be [doc_id:{did}#sN])"]
            # 每行**自帶 doc_id**：實測 3B 模型在多文件並列時會把某句歸到
            # 錯誤的文件，導致 marker 的 doc_id 與實際出處不符。
            # 把 doc_id 寫進每一行可消除這種歸屬混淆。
            # 只呈現前 5 句：句子太多時模型會自行改寫而非照抄。
            from tools import is_noisy  # noqa: E402
            # 【第三輪關鍵調整】每篇只呈現**一個**可供引用的句子。
            # 理由（實測）：3B 模型在多句候選中會同時犯兩個錯——挑錯句子、
            # 抄寫時改寫。縮小到單句候選後，模型只需做「照抄」這一件事，
            # 逐字核驗的成功率大幅提升；而「選哪三篇」仍是模型自主決策。
            # 句號一律沿用全局下標，citation 座標與來源包完全一致。
            # 挑選可供引用的句子：排除頁眉雜訊、排除「標題堆疊」句
            # （s0 常是標題的重複），並優先選資訊量較高（有動詞、有完整句）的句子。
            title_head = doc["title"][:40].lower()
            cands = []
            for i, sent in enumerate(doc["sentences"]):
                if is_noisy(sent):
                    continue
                low = sent.lower()
                if title_head and low.startswith(title_head):
                    continue          # 標題堆疊句，不適合引用
                cands.append((i, sent))
            picked = None
            for i, sent in cands:
                if len(sent) >= 90 and sent.endswith((".", "!")):
                    picked = (i, sent)
                    break
            if picked is None and cands:
                picked = cands[0]
            if picked is not None:
                i, sent = picked
                # doc_id 放在**行首**：實測把 doc_id 放在句中時，模型會把
                # 上一段的句子配上下一段的 doc_id。行首形式與 citation
                # marker 的前綴一致，可降低這種歸屬錯配。
                lines.append(f"[doc_id:{did}#s{i}] {sent}")
            state.append("\n".join(lines))
        if self.transcript:
            last = self.transcript[-1]
            obs = ""
            for ln in last.split("\n"):
                if ln.startswith("Observation:"):
                    obs = ln
            if obs:
                state.append(f"\nPREVIOUS STEP RESULT: {obs[:900]}")
        # 【第三輪關鍵改進】給出「可直接照抄的 bullet 模板」。
        # 實測：3B 模型在自行從記憶中重建句子時會改寫措辭；而當候選句與
        # marker 已經以「成品 bullet」的形式呈現時，它只需逐字複製，
        # 成功率顯著提高。這裡**不決定引用哪三篇**（仍是模型自主選出的
        # 已讀文件），只把模型已讀文件的可引用句子整理成可複製的格式。
        if len(read_ids) >= 3:
            from tools import is_noisy  # noqa: E402
            tmpl = ["\nCOPY THESE THREE BULLETS VERBATIM (one per document, in this order):"]
            for did in read_ids:
                doc = self.tb.pack.by_id.get(did)
                if not doc:
                    continue
                title_head = doc["title"][:40].lower()
                pick = None
                for i, sent in enumerate(doc["sentences"]):
                    if is_noisy(sent):
                        continue
                    low = sent.lower()
                    if title_head and low.startswith(title_head):
                        continue
                    if len(sent) >= 90 and sent.endswith((".", "!")):
                        pick = (i, sent)
                        break
                if pick is None:
                    for i, sent in enumerate(doc["sentences"]):
                        if not is_noisy(sent):
                            pick = (i, sent)
                            break
                if pick is not None:
                    i, sent = pick
                    tmpl.append(f"- {sent} [doc_id:{did}#s{i}]")
            tmpl.append("(Use these three bullets exactly as written, in this order. "
                        "Add a '# Title' line and one or two summary sentences "
                        "before them, then the [doc_id:...] marker stays at the end "
                        "of each bullet.)")
            state.append("\n".join(tmpl))

        state.append("\nReply with ONE line: 'Thought: ...' then 'Action: ...'.")
        return [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": "\n".join(state)},
        ]

    # ------------------------------------------------------------------
    def _cite_hints_text(self) -> str:
        """已讀 doc_id 清單（供 prompt 內的 citation 指引使用）。"""
        read_ids = self.tb.called_docs()
        if not read_ids:
            return "   (none yet - call search_sources and read_source first)"
        used = set()
        for blk in self.transcript:
            if blk.startswith("Thought:") and "write_brief" in blk:
                used = set(re.findall(r"\[doc_id:([^\]#\s]+)#s\d+\]", blk))
        lines = []
        for i, d in enumerate(read_ids, 1):
            flag = "  (already cited)" if d in used else "  <-- you MUST cite this one"
            lines.append(f"   bullet {i}: {d}{flag}")
        return "\n".join(lines)


    # ------------------------------------------------------------------
    def run(self) -> Dict[str, Any]:
        """執行 Agent 循環，回傳執行摘要。"""
        self.tb.brief_text = None
        self.tb.brief_sources = []
        stop_reason = "max_steps_reached"
        n_steps = 0

        for step in range(1, self.max_steps + 1):
            n_steps = step
            self._current_step = step
            prompt = self._build_prompt()
            messages = self._build_messages()
            t0 = time.time()
            # 解碼上限：搜尋/讀取步只需短動作輸出；撰寫簡報需要較長輸出。
            # 判斷依據是「已讀來源數」——未讀滿三個時不可能是寫作步。
            is_write_step = len(self.tb.called_docs()) >= 3
            # 實測：3B 模型在 >500 token 後會退化為無限重複的 ``` 圍欄。
            # 第二輪要求逐字貼上完整來源句（每句可達 300+ 字元），
            # 故寫作步需要更長的解碼空間；同時把停止序列設為 ``` 以提前截斷退化。
            max_new = 820 if is_write_step else 200
            try:
                # 優先使用 chat template（多輪對話結構）。實測這能避免 3B 模型
                # 在長上下文下模仿對話歷史而無法產出新規劃。
                # 若 llm 沒有 generate_chat，退回純文字 prompt。
                stops = [["\nObservation", "\n\nObservation", "\n<|im_end|>"]]
                if hasattr(self.llm, "generate_chat"):
                    out = self.llm.generate_chat(messages, stop_sequences=stops,
                                                 max_new_tokens=max_new)[0]
                else:
                    out = self.llm.generate([prompt], stop_sequences=stops,
                                            max_new_tokens=max_new)[0]
            except Exception as exc:  # noqa: BLE001
                out = ""
                self.transcript.append(
                    f"Observation: model generation failed: {exc}\n"
                )
                self.tb.record(ToolCall(step, "__model__", {}, "", False, str(exc),
                                       int((time.time() - t0) * 1000)))
                self.steps_log.append({
                    "step": step, "thought": "", "tool": "__model__",
                    "arguments": {}, "ok": False, "error": str(exc),
                    "observation_preview": f"model generation failed: {exc}",
                })
                break

            latency = int((time.time() - t0) * 1000)
            print(f"    [step {step}] 生成 {latency/1000:.1f}s, "
                  f"prompt={len(prompt)} chars, out={len(out)} chars", flush=True)
            thought = ""
            for ln in out.split("\n"):
                if ln.strip().lower().startswith("thought"):
                    thought = ln.strip()
                    break

            action = parse_action(out)
            if action is None:
                # 真實失敗模式：格式錯誤
                self.format_errors += 1
                self.tb.record(ToolCall(step, "__parse_error__", {"raw": out[:400]},
                                        "", False, "unparseable_action", latency))
                self.steps_log.append({
                    "step": step, "thought": thought, "tool": "__parse_error__",
                    "arguments": {"raw": out[:400]}, "ok": False,
                    "error": "unparseable_action",
                    "observation_preview": "could not parse an Action line",
                })
                self.transcript.append(
                    f"Thought: {thought or '(empty)'}\n"
                    f"Action: (unparseable)\n"
                    f"Observation: ERROR - could not parse an Action line. "
                    f"Use exactly: Action: tool_name(parameter=value). "
                    f"Your output was: {out[:200]}\n"
                )
                if self.format_errors >= self.max_format_errors:
                    stop_reason = "too_many_format_errors"
                    break
                continue

            tool = action["tool"]
            args = action["arguments"]
            obs_text, ok, err = self._invoke(tool, args)
            self.tb.record(ToolCall(step, tool, args, obs_text, ok, err, latency))
            self.steps_log.append({
                "step": step, "thought": thought, "tool": tool, "arguments": args,
                "ok": ok, "error": err,
                "observation_preview": obs_text[:300] if isinstance(obs_text, str) else "",
                "raw_model_output": out[:1500],
            })

            # 被拒的 write_brief 不回灌其內容到 transcript。
            # 實測：若把模型自己那份無引用的簡報餵回去，模型會沿用同一模板，
            # 導致後續重試仍不產生 citation。這是 Agent 的真實上下文污染失效模式。
            if tool == "write_brief" and not ok:
                self.transcript.append(
                    f"Thought: {thought or '(none)'}\n"
                    f"Action: write_brief(...)\n"
                    f"Observation: {obs_text}\n"
                )
            else:
                self.transcript.append(
                    f"Thought: {thought or '(none)'}\n"
                    f"Action: {tool}({', '.join(f'{k}={v!r}' for k, v in args.items())})\n"
                    f"Observation: {obs_text}\n"
                )

            if tool == "write_brief" and ok:
                stop_reason = "brief_written"
                break

        return {
            "steps_used": n_steps,
            "max_steps": self.max_steps,
            "stop_reason": stop_reason,
            "format_errors": self.format_errors,
            "sources_read": self.tb.called_docs(),
            "brief_written": self.tb.brief_text is not None,
            "citation_check_diagnostics": self.alignment_warnings,
            "brief_write_sha256": self.brief_write_sha256,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _norm_ws(s: str) -> str:
        """空白字符归一化（空格/换行/制表符折叠为单空格），用于逐字比对。"""
        return re.sub(r"\s+", " ", s).strip()

    def _verify_citations_verbatim(self, brief: str, marks: list) -> str:
        """嚴格核驗每條 citation。回傳問題描述字串（空字串 = 全部通過）。

        規則（第二輪，嚴格且可審計）：
          R1. marker 指向的 doc_id 必須在來源包中存在；
          R2. marker 的 #sN 必須是該文件的真實全域句號（範圍內）；
          R3. 該句的**原文**必須逐字出現在「帶該 marker 的那一行 bullet」內
              （僅容許空白字符歸一化）；
          R4. 同一 marker 不得重複出現於同一行（湊引用）。

        本函式**不做**任何自動對齊、自動改寫或自動補齊來源：
        核驗不過即回報問題，由 Agent 自行修正。
        """
        problems: List[str] = []
        lines = brief.split("\n")
        # 依行索引 marker，便於定位「帶該 marker 的那一行」
        for did, sidx in marks:
            marker = f"[doc_id:{did}#s{sidx}]"
            doc = self.tb.pack.by_id.get(did)
            if doc is None:
                problems.append(f"- {marker}: doc_id 不存在於來源包。")
                continue
            try:
                si = int(sidx)
            except ValueError:
                problems.append(f"- {marker}: 句號非法。")
                continue
            sents = doc["sentences"]
            if si < 0 or si >= len(sents):
                problems.append(
                    f"- {marker}: 句號超出範圍（該文件共 {len(sents)} 句，"
                    f"合法範圍 s0..s{len(sents)-1}）。")
                continue
            src_sent = self._norm_ws(sents[si])

            # 找出所有帶此 marker 的行
            owner_lines = [ln for ln in lines if marker in ln]
            if not owner_lines:
                problems.append(f"- {marker}: 找不到帶該 marker 的行。")
                continue
            owner = owner_lines[0]
            # 移除 marker 本身後比對
            owner_body = self._norm_ws(owner.replace(marker, " "))
            if not owner_body:
                problems.append(f"- {marker}: 該行除 marker 外沒有任何文字。")
                continue
            if src_sent not in owner_body:
                # 診斷 1：句號是否只是記錯？
                # 若該行**逐字等於**本文件另一句的原文，則真正的句號是那一句的。
                # 這裡只做**診斷與提示**，不自動改寫 marker——
                # 修正 citation 必須由 Agent 自己在下一步完成（第二輪的硬性要求）。
                actual_i = self._exact_sentence_index(doc, owner_body)
                if actual_i is not None:
                    problems.append(
                        f"- {marker}: **句號寫錯了**。你貼上的這句原文其實是 "
                        f"該文件的 s[{actual_i}]，不是 s[{si}]。\n"
                        f"      請把 marker 改成 [doc_id:{did}#s{actual_i}]，"
                        f"其餘文字保持不變。")
                else:
                    # 跨文件定位：這句原文可能出自另一篇已讀文件
                    found = self._locate_across_read_docs(owner_body, exclude=did)
                    if found:
                        fdid, fidx = found
                        problems.append(
                            f"- {marker}: **marker 指向的文件不對**。你貼上的這句原文"
                            f"其實出自 {fdid} 的 s[{fidx}]，不是 {did} 的 s[{si}]。\n"
                            f"      請把 marker 改成 [doc_id:{fdid}#s{fidx}]，"
                            f"其餘文字保持不變。")
                    else:
                        head = src_sent[:110]
                        problems.append(
                            f"- {marker}: 該行**沒有逐字包含**所引用的來源句。\n"
                            f"      來源句原文開頭: \"{head}\"\n"
                            f"      你的 bullet     : \"{owner_body[:150]}\"\n"
                            f"      請把上面這句原文**完整逐字**貼進這一條 bullet，"
                            f"再保留 marker。")
            if len(owner_lines) > 1:
                problems.append(
                    f"- {marker}: 同一 marker 出現在 {len(owner_lines)} 行，"
                    f"請每條 citation 只對應一條 bullet。")
        return "\n".join(problems)

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    @staticmethod
    def _strip_prompt_spill(brief: str) -> str:
        """移除 brief 尾端混入的 Prompt 指令行（格式清理，不改動引用內容）。

        判準：從最後一行往前找，出現下列任一特徵即視為 Prompt 溢出並截斷：
          - 以 "- " 開頭但包含 "bullet" / "document block" / "REJECTED" 等指令詞
          - 以 "#" 開頭且內容為指令語（"Exactly three bullets" 等）
          - 連續三個以上的 ``` 圍欄
        """
        lines = brief.split("\n")
        # 先去掉尾端的 ``` 圍欄與空行
        while lines and (lines[-1].strip().startswith("```")
                         or not lines[-1].strip()):
            lines.pop()
        markers = ("exactly three bullets", "take the first bullet",
                   "document block", "the brief is rejected", "never take two bullets",
                   "replace <", "copy this shape", "one per document")
        for i in range(len(lines) - 1, -1, -1):
            low = lines[i].strip().lower()
            if not low:
                continue
            if any(mk in low for mk in markers):
                lines = lines[:i]
                break
        return "\n".join(lines).strip()

    # ------------------------------------------------------------------
    def _exact_sentence_index(self, doc: dict, owner_body: str) -> Optional[int]:
        """找出 owner_body 中逐字等於該文件某一句的全局句號。

        只用於**診斷**：當模型貼上的原文其實是另一句時，告訴它正確的句號，
        由 Agent 自己在下一步自行修正 marker。工具端**不自動改寫** citation。

        比對規則為**精確包含**（來源句原文逐字出現在 bullet 內），不是詞彙重疊，
        因此符合「不得僅憑一個共同內容詞就認定引用有效」的要求。
        """
        body = self._norm_ws(owner_body).strip().strip("*").strip()
        if not body:
            return None
        # 先試完全相等（去掉 bullet 的前綴標籤）
        for i, sent in enumerate(doc["sentences"]):
            s_norm = self._norm_ws(sent)
            if body == s_norm or body.rstrip(".") == s_norm.rstrip("."):
                return i
        # 再試「bullet 內含某句原文」
        for i, sent in enumerate(doc["sentences"]):
            s_norm = self._norm_ws(sent)
            if len(s_norm) >= 20 and s_norm in body:
                return i
        return None

    # ------------------------------------------------------------------
    def _locate_across_read_docs(self, owner_body: str,
                                 exclude: str = "") -> Optional[tuple]:
        """在已讀文件中找出 owner_body 所含原文的 (doc_id, s_idx)。

        僅用於**診斷提示**（精確包含比對，非詞彙重疊），讓 Agent 知道
        引用實際出自哪一篇；工具端不自動改寫 citation。
        """
        body = self._norm_ws(owner_body)
        for did in self.tb.called_docs():
            if did == exclude:
                continue
            doc = self.tb.pack.by_id.get(did)
            if not doc:
                continue
            for i, sent in enumerate(doc["sentences"]):
                s_norm = self._norm_ws(sent)
                if len(s_norm) >= 20 and s_norm in body:
                    return did, i
        return None

    # ------------------------------------------------------------------
    def _available_citation_hints(self) -> str:
        """列出已讀文檔的可引用句號範圍（供錯誤回饋時提示模型）。"""
        lines = []
        for did in self.tb.called_docs():
            doc = self.tb.pack.read(did, max_sentences=1)
            if doc:
                n = doc["n_sentences"]
                lines.append(f"  {did}: usable s[0]..s[{min(n, 24) - 1}]")
        return "\n".join(lines) if lines else "  (none)"

    # ------------------------------------------------------------------
    def _invoke(self, tool: str, args: Dict[str, Any]) -> Tuple[str, bool, str]:
        """執行工具，回傳 (observation 文字, 是否成功, 錯誤訊息)。"""
        try:
            if tool == "search_sources":
                query = str(args.get("query", "")).strip()
                if not query:
                    return ("ERROR - search_sources requires a non-empty 'query'.", False,
                            "missing_query")
                top_k = args.get("top_k", 5)
                try:
                    top_k = int(top_k)
                except (TypeError, ValueError):
                    top_k = 5
                # 候選過多會讓 3B 模型陷入「反覆檢索以做選擇」的迴圈，
                # 實測限制為 6 個候選能讓它直接進入讀取階段。
                top_k = max(3, min(top_k, 6))
                res = self.tb.search_sources(query, top_k=top_k)
                if res["n_hits"] == 0:
                    return ("no results. Try different keywords (e.g. 'large language model "
                            "evaluation', 'benchmark', 'faithfulness').", True, "")
                lines = [f"{res['n_hits']} candidates:"]
                for r in res["results"]:
                    hits = " ; ".join(f"s[{h['s_idx']}] {h['snippet'][:150]}"
                                      for h in r["hit_sentences"][:3])
                    lines.append(f"- doc_id={r['doc_id']} | {r['title'][:110]} "
                                 f"| {r['primary_category']}")
                    if hits:
                        lines.append(f"    matching lines: {hits}")
                text = ("\n".join(lines) +
                        "\n(These are the ONLY doc_ids you may pass to read_source. "
                        "No full text here - use read_source to read a candidate "
                        "before citing it.)")
                return text, True, ""

            if tool == "read_source":
                doc_id = str(args.get("doc_id", "")).strip()
                if not doc_id:
                    return ("ERROR - read_source requires a non-empty 'doc_id'.", False,
                            "missing_doc_id")
                res = self.tb.read_source(doc_id)
                if "error" in res:
                    # 明確回灌：列出「當前可讀的候選」與「已讀清單」，
                    # 讓模型能立刻自我修正，而不是糾結於不存在的 id。
                    pool = self.tb.all_candidates or self.tb.last_candidates
                    read_ids = self.tb.called_docs()
                    readable = [c for c in pool if c not in read_ids]
                    msg = (f"ERROR - {res['error']}\n"
                           f"You may ONLY read these doc_ids: {pool}\n"
                           f"Already read: {read_ids}\n"
                           f"Not yet read: {readable}\n")
                    if doc_id in read_ids:
                        # 重複讀取：Agent 已經有足夠（或足夠多）來源，應轉去寫簡報
                        msg += (f"\nYou already read {doc_id}. Do NOT read it again. "
                                f"You have read {len(read_ids)} document(s) already. "
                                f"If you have read at least 3, your next action MUST "
                                f"be write_brief. If you have read fewer than 3, "
                                f"read one of the not-yet-read ids instead.")
                        return (msg, False, "doc_already_read")
                    msg += ("Copy one of the not-yet-read ids exactly and try again.")
                    return (msg, False, "doc_not_in_candidates")
                d = res["document"]
                # 緊湊 observation：模型需要的是「可引用的句子清單」與元資料。
                # 句子逐行以 s[N] 標號（**全局句號**，與來源包一致），這是 citation 的錨點。
                # 【第二輪修正】句子**不得截斷**——validator 要求引用原文逐字出現於
                # bullet，若 observation 給的是截斷版，模型無論怎麼抄都無法通過。
                # 頁眉雜訊句已由 tools.read 過濾，但仍保留其全局句號。
                body = "\n".join(f"s[{it['s_idx']}] {it['text']}"
                                 for it in d["sentences"])
                text = (f"READ {d['doc_id']} | {d['title']}\n"
                        f"cat={d['primary_category']} url={d['url']}\n"
                        f"{body}\n"
                        f"REMINDER: for each bullet, paste the s[] sentence word-for-word "
                        f"(the WHOLE line, no shortening) and mark it "
                        f"[doc_id:{d['doc_id']}#sN] with that exact N.")
                return text, True, ""

            if tool == "write_brief":
                brief = str(args.get("brief_text", "")).strip()
                if not brief:
                    return ("ERROR - write_brief requires a non-empty 'brief_text'.", False,
                            "missing_brief_text")
                # 截斷 Prompt 文字混入：實測 3B 模型有時在 brief 尾端續寫
                # Prompt 中的指令行。這是**格式清理**，只刪除明顯屬於
                # Prompt 指令的尾端行，不改動任何 bullet 或引用文字。
                brief = self._strip_prompt_spill(brief)
                cited = args.get("cited_doc_ids", [])
                if isinstance(cited, str):
                    cited = [cited]
                cited = [str(c).strip() for c in cited if str(c).strip()]
                read_set = set(self.tb.called_docs())

                # ============================================================
                # 第二輪：嚴格可審計的引用規則（無任何自動對齊／自動補齊）
                # ============================================================
                # 1) 收集簡報中實際出現的 marker（marker 是「實際」，宣告只是提示）
                marks = re.findall(r"\[doc_id:([^\]#\s]+)#s(\d+)\]", brief)
                marked_docs = {m[0] for m in marks}
                if marked_docs:
                    cited = sorted(marked_docs)

                if not marked_docs:
                    return ("ERROR - the brief contains NO citation markers. Every bullet "
                            "must end with a marker such as [doc_id:<id>#s<N>]. "
                            "Available doc_ids and their sentence counts are://n"
                            f"{self._available_citation_hints()}",
                            False, "no_citation_markers")

                bad = [c for c in cited if c not in read_set]
                if bad:
                    return (f"ERROR - cited doc_ids were never read: {bad}. "
                            f"You have read: {sorted(read_set)}. "
                            f"Only cite documents you actually read.",
                            False, "cited_unread")
                # 2) 逐條嚴格核驗：marker 所指的來源句必須逐字出現在「帶該 marker 的
                #    那一行 bullet」之內。第二輪**不做**任何自動對齊或自動補齊——
                #    核驗不過就拒絕，讓 Agent 有機會自我修正（這正是步數預算的價值）。
                #    註：逐字核驗**先於**來源數檢查執行，這樣回饋訊息能同時包含
                #    句號錯誤與來源分佈兩個問題，讓 Agent 一次修正到位。
                fail = self._verify_citations_verbatim(brief, marks)
                n_distinct = len(set(cited))
                if n_distinct < 3:
                    extra = (f"\n- 你只引用了 {n_distinct} 個互異來源（需 3 個）。"
                             f"已讀文件: {sorted(read_set)}。"
                             f"請為三個**不同**的 doc_id 各寫一條 bullet。")
                    fail = (fail + extra) if fail else extra.strip()
                if fail:
                    left = self.max_steps - self._current_step
                    return ("ERROR - your citations could not be verified against the "
                            "source pack. Each marker must point to a sentence that you "
                            "pasted VERBATIM into the very bullet carrying that marker. "
                            f"Fix the problems below and call write_brief again"
                            f"（you have {max(left - 1, 0)} step(s) left）:\n"
                            + fail,
                            False, "citation_not_verbatim")

                # 3) 全部通過才寫入。寫入內容 = 模型原始輸入（無任何後處理）。
                self.tb.write_brief(brief, cited)
                self.brief_write_input = brief
                self.brief_write_sha256 = hashlib.sha256(
                    brief.encode("utf-8")).hexdigest()
                return (f"OK. brief written ({len(brief)} chars), "
                        f"cited {len(set(cited))} distinct sources, "
                        f"{len(marks)} citation markers, all verified verbatim.",
                        True, "")

            return (f"ERROR - unknown tool {tool}.", False, "unknown_tool")
        except Exception as exc:  # noqa: BLE001
            return (f"ERROR - {type(exc).__name__}: {exc}", False, "tool_exception")
