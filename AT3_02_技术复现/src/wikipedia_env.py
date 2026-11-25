"""
wikipedia_env.py — ReAct 的 Wikipedia 交互环境（从零实现）

论文 §3.1 对三动作的定义：
  (1) search[entity]  —— 返回对应实体 wiki 页的**前 5 句**；若實體不存在，则返回
                         搜索引擎的 **top-5 相似实体**。
  (2) lookup[string]  —— 返回**页面中包含该 string 的下一句**，模拟浏览器 Ctrl+F。
  (3) finish[answer]  —— 结束当前任务并给出答案。

官方实现 wikienv.py 补充了论文未写明的行为细节（本次复现照此实现）：
  - search 成功：页面按段落切 -> 按句切 -> 取前 5 句，以空格拼接返回
  - search 失败：f"Could not find {entity}. Similar: {titles[:5]}."
  - lookup 返回：f"(Result {cnt+1} / {len(list)}) " + sentence   （带进度前缀）
  - lookup 越界："No more results.\n"
  - lookup 游标：同一 keyword 连续 lookup 递增游标返回第 2、3 个结果；换 keyword 重置
  - finish：f"Episode finished, reward = {reward}\n"，done=True
  - 非法动作：f"Invalid action: {action}"
  - think[...]：返回 "Nice thought."（保留为合法动作但不改变环境 —— 这正是
    论文中「thought 在机制上不可能改变环境」的技术体现）

【本次复现的偏离声明】
原论文环境直接抓取 en.wikipedia.org。本机网络环境中该主站不可达，故改用
`api.wikimedia.org` 的官方 REST API 重建，语义保持等价：
  - search  -> GET /core/v1/wikipedia/en/search/page?q={entity}&limit=6
               若结果中存在标题精确匹配（不区分大小写）则取该页正文前 5 句；
               否则返回 top-5 相似实体标题。
  - lookup  -> 在 search 返回的同一页正文中做 Ctrl+F 语义查找。
  - page    -> GET /core/v1/wikipedia/en/page/{title}  取 wikitext，转换为纯文本句子。
所有抓取结果落盘缓存，避免重复请求并保证同一次实验内环境一致（论文附录 A.2
已指出 Wikipedia 页面会随时间变化，影响可复现性）。
"""
from __future__ import annotations

import json
import re
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

# ---------------------------------------------------------------------------
# wikitext -> 纯文本
# ---------------------------------------------------------------------------
_RE_REF = re.compile(r"\[\[([^\]|]*\|)?([^\]]*)\]\]")       # [[a|b]] -> b
_RE_REF_NOLINK = re.compile(r"\[\[([^\]]*)\]\]")            # [[a]] -> a
_RE_TAG = re.compile(r"<[^>]+>")                             # <ref>...</ref> 等
_RE_EXTERNAL = re.compile(r"\[https?://\S+\s+([^\]]*)\]")   # [url text]
_RE_FILE = re.compile(r"\[\[(?:File|Image|文件|圖像):[^\]]*\]\]", re.IGNORECASE)


def _strip_templates(text: str) -> str:
    """移除 {{...}} 模板，支持嵌套。

    简单的非贪婪正则无法处理嵌套模板（如 {{infobox {{nested}} }}），
    这里用括号计数做逐层剥离。
    """
    while True:
        start = text.find("{{")
        if start == -1:
            return text
        depth = 0
        end = -1
        for i in range(start, len(text) - 1):
            if text[i:i + 2] == "{{":
                depth += 1
                i += 1
            elif text[i:i + 2] == "}}":
                depth -= 1
                if depth == 0:
                    end = i + 2
                    break
        if end == -1:
            # 括号不配对，丢弃剩余部分
            return text[:start]
        text = text[:start] + " " + text[end:]


def wikitext_to_text(src: str) -> str:
    """把 wikitext 粗略转成可切句的纯文本。目标是语义可用，而非完美还原。"""
    text = src
    text = _RE_FILE.sub(" ", text)
    text = _RE_EXTERNAL.sub(r"\1", text)
    text = _strip_templates(text)
    text = _RE_TAG.sub(" ", text)
    text = _RE_REF.sub(r"\2", text)
    text = _RE_REF_NOLINK.sub(r"\1", text)
    text = re.sub(r"'''?", "", text)                          # 粗体/斜体标记
    text = text.replace("&nbsp;", " ")
    # 段落分隔：官方环境按 '\n' 切段落，因此先把段落标记保留下来
    text = re.sub(r"\n+", "\n", text)
    return text


def split_sentences(text: str) -> List[str]:
    """按句切分。官方实现按 '. ' 切并给每句补 '.'。"""
    sentences: List[str] = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        # 按句末标点切分，保留标点
        for raw in re.split(r"(?<=[.!?])\s+", para):
            s = raw.strip()
            if s:
                sentences.append(s if s[-1] in ".!?" else s + ".")
    return sentences


# ---------------------------------------------------------------------------
# 环境
# ---------------------------------------------------------------------------
@dataclass
class Observation:
    text: str
    done: bool = False
    reward: float = 0.0
    info: Dict[str, str] = field(default_factory=dict)


class WikipediaEnv:
    """ReAct 的 Wikipedia 环境（search / lookup / finish）。"""

    API_BASE = "https://api.wikimedia.org/core/v1/wikipedia/en"

    def __init__(
        self,
        cache_dir: str,
        timeout: float = 20.0,
        max_retries: int = 3,
        user_agent: str = "AT3-ReAct-Reproduction/1.0",
        search_sentences: int = 5,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.max_retries = max_retries
        self.search_sentences = search_sentences
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})

        # 每次 step 之间重置的状态（对应官方 wikienv 的实例属性）
        self.current_page_text: str = ""
        self.current_page_title: str = ""
        self.lookup_cache: Dict[str, List[str]] = {}
        self.lookup_cursor: Dict[str, int] = {}

    # ---------------- 缓存与抓取 ----------------
    def _cache_path(self, kind: str, key: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", key)[:120]
        return self.cache_dir / f"{kind}__{safe}.json"

    def _cached_get(self, kind: str, key: str, url: str) -> Optional[dict]:
        path = self._cache_path(kind, key)
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return None
        data = self._fetch(url)
        if data is not None:
            try:
                path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            except OSError:
                pass
        return data

    def _fetch(self, url: str) -> Optional[dict]:
        """抓取並解析 JSON。單次請求有硬性時間上限，避免整個實驗被掛起。"""
        for attempt in range(self.max_retries):
            try:
                resp = self.session.get(url, timeout=self.timeout)
                if resp.status_code == 200:
                    return resp.json()
                # 404 是确定性结果，不重试
                if resp.status_code == 404:
                    return None
            except (requests.RequestException, ValueError):
                pass
            if attempt < self.max_retries - 1:
                time.sleep(1.5 * (attempt + 1))
        return None

    # ---------------- 页面内容 ----------------
    def _page_text(self, title: str) -> Optional[str]:
        """取 wiki 页纯文本。title 需为 URL 编码后的路径片段。"""
        url = f"{self.API_BASE}/page/{urllib.parse.quote(title.replace(' ', '_'), safe='')}"
        data = self._cached_get("page", title, url)
        if data is None:
            return None
        src = data.get("source") or data.get("html") or ""
        if not src:
            return None
        return wikitext_to_text(src)

    # ---------------- 回合状态 ----------------
    def reset_episode(self) -> None:
        """重置逐回合状态。

        环境的 current_page_text / lookup_cache / lookup_cursor 是**逐题**的，
        新的一题开始前必须清空，否则本题的 lookup 会作用在上一题的页面上。
        """
        self.current_page_text = ""
        self.current_page_title = ""
        self.lookup_cache = {}
        self.lookup_cursor = {}

    # ---------------- 三种动作 ----------------
    def search(self, entity: str) -> str:
        """search[entity]：页面前 5 句；查无则给 top-5 相似实体。"""
        entity = entity.strip()
        if not entity:
            return "Could not find . Similar: []."

        url = (
            f"{self.API_BASE}/search/page?q={urllib.parse.quote(entity)}&limit=6"
        )
        data = self._cached_get("search", entity, url)
        titles: List[str] = []
        if data:
            for p in data.get("pages", []):
                t = p.get("title")
                if t:
                    titles.append(t)

        # 标题精确匹配（不区分大小写）才算命中，否则按原论文语义返回 top-5 相似实体
        hit = None
        for t in titles:
            if t.lower() == entity.lower():
                hit = t
                break

        if hit is not None:
            text = self._page_text(hit)
            if text:
                sents = split_sentences(text)[: self.search_sentences]
                if sents:
                    self.current_page_text = text
                    self.current_page_title = hit
                    self.lookup_cache = {}
                    self.lookup_cursor = {}
                    return " ".join(sents)
            # 正文取不到则退化为用 search 的 excerpt
            excerpt = ""
            if data:
                for p in data.get("pages", []):
                    if p.get("title") == hit:
                        excerpt = re.sub(r"<[^>]+>", "", p.get("excerpt") or "")
                        break
            if excerpt:
                self.current_page_text = ""
                self.current_page_title = hit
                self.lookup_cache = {}
                self.lookup_cursor = {}
                return excerpt

        similar = titles[:5]
        return f"Could not find {entity}. Similar: {similar}."

    def lookup(self, string: str) -> str:
        """lookup[string]：页面中包含该 string 的句子，带 (Result i / n) 前缀。"""
        string = string.strip()
        if not string:
            return "No more results.\n"
        if not self.current_page_text:
            return "No more results.\n"

        if string not in self.lookup_cache:
            # 官方语义：返回包含该 string 的句子列表（大小写不敏感更符合 Ctrl+F）
            hits = [s for s in split_sentences(self.current_page_text)
                    if string.lower() in s.lower()]
            self.lookup_cache[string] = hits
            self.lookup_cursor[string] = 0
        hits = self.lookup_cache[string]
        cur = self.lookup_cursor.get(string, 0)
        if cur < len(hits):
            self.lookup_cursor[string] = cur + 1
            return f"(Result {cur + 1} / {len(hits)}) " + hits[cur]
        return "No more results.\n"

    def finish(self, answer: str) -> str:
        return f"Episode finished, reward = 1\n"

    # ---------------- 主入口 ----------------
    def step(self, action: str) -> Observation:
        """执行一个动作。action 形如 'search[Foo]' / 'lookup[bar]' / 'finish[ans]'。

        对应官方 wikienv.py 的 step()。注意环境**不接受 thought**：
        think[...] 返回 "Nice thought."，不改变任何状态。
        """
        action = (action or "").strip()
        action = action.replace("\n", " ")

        if action.startswith("think["):
            return Observation(text="Nice thought.", done=False, reward=0.0)

        m = re.match(r"^(\w+)\[(.*)\]$", action, flags=re.DOTALL)
        if not m:
            return Observation(text=f"Invalid action: {action}", done=False, reward=0.0)

        name, arg = m.group(1).lower(), m.group(2)
        if name == "search":
            return Observation(text=self.search(arg), done=False, reward=0.0)
        if name == "lookup":
            return Observation(text=self.lookup(arg), done=False, reward=0.0)
        if name == "finish":
            return Observation(text=self.finish(arg), done=True, reward=1.0,
                               info={"answer": arg})
        return Observation(text=f"Invalid action: {action}", done=False, reward=0.0)
