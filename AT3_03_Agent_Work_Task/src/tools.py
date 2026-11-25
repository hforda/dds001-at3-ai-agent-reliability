"""
tools.py — AT3 ③ 的三個核心工具（純本地、離線、標準庫）

課程要求的工具集合（不可增減核心三項）：
  1. search_sources : 在本地來源包中檢索相關文檔
  2. read_source    : 讀取指定文檔，並保留實際閱讀的內容及位置
  3. write_brief    : 根據已讀取的來源生成研究簡報

設計原則：
  - **全部唯讀或本地寫入**，不觸網、不呼叫 API、不呼叫外部模型。
  - search_sources 只回傳「索引層」資訊（標題、類別、命中句號與片段），
    **不回傳全文**，以模擬真實檢索工具並迫使 Agent 必須呼叫 read_source。
  - read_source 回傳全文句子（帶全域句號 s_idx），這是 citation 的回查錨點。
  - 每次工具呼叫都會記錄 observation，供 trajectory / tool-results 交叉核對。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# 來源包載入與檢索
# ---------------------------------------------------------------------------
_STOP = {
    "the", "a", "an", "of", "and", "or", "in", "on", "for", "to", "is", "are",
    "with", "by", "that", "this", "we", "it", "as", "be", "from", "at", "has",
    "can", "which", "their", "its", "not", "but", "how", "what", "when", "there",
    "these", "those", "our", "more", "than", "also", "been", "such", "using",
    "based", "show", "shows", "results", "result", "paper", "work", "propose",
    "proposed", "approach", "method", "methods", "task", "tasks", "model", "models",
    "data", "use", "used", "using", "two", "one", "three", "may", "we", "our",
}


# arXiv 轉檔的前幾句常含頁眉雜訊（作者郵箱、日期分隔線、LaTeX 殘留）。
# 這些句子不適合被引用，但它們**仍是來源包的一部分**，句號必須保持不變，
# 否則 citation 座標會與來源包錯位。因此：只在「呈現給模型」時過濾，
# 句號一律沿用全局下標。
_NOISE_PATTERNS = [
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),  # email
    re.compile(r"={10,}"),                       # LaTeX 分隔線
    re.compile(r"\[\s*\[\s*\[?\s*\d{4}"),  # arXiv 日期戳
    re.compile(r"\[NO "),                      # 缺失標題/作者標記
    re.compile(r"\bAbstract\b\s*$"),
]


def is_noisy(sentence: str) -> bool:
    """判斷一句是否為 arXiv 頁眉雜訊（不適合引用，但句號仍保留）。"""
    s = sentence.strip()
    if not s:
        return True
    if len(s) < 25:
        # 過短的片段（如 "Wu et al."、"N."）多為作者署名殘留
        return True
    return any(p.search(s) for p in _NOISE_PATTERNS)


def tokenize(text: str) -> List[str]:
    """把查詢字串切成檢索詞。"""
    words = re.findall(r"[a-z0-9\-]+", text.lower())
    return [w for w in words if w not in _STOP and len(w) > 2]


class SourcePack:
    """本地離線來源包（記憶體索引 + 檔案查證）。"""

    def __init__(self, pack_dir: Path) -> None:
        self.pack_dir = Path(pack_dir)
        self.docs_path = self.pack_dir / "documents.jsonl"
        self.index_path = self.pack_dir / "index.json"
        if not self.docs_path.exists():
            raise FileNotFoundError(f"來源包不存在：{self.docs_path}")
        self.docs: List[Dict[str, Any]] = []
        with self.docs_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.docs.append(json.loads(line))
        self.by_id: Dict[str, Dict[str, Any]] = {d["doc_id"]: d for d in self.docs}
        self.index = json.loads(self.index_path.read_text(encoding="utf-8"))

        # 檢索索引：title tokens + 首 12 句 tokens（摘要層）
        self._title_tok: List[set] = []
        self._head_tok: List[set] = []
        for d in self.docs:
            self._title_tok.append(set(tokenize(d["title"])))
            head = " ".join(d["sentences"][:12])
            self._head_tok.append(set(tokenize(head)))
        print(f"[SourcePack] 載入 {len(self.docs)} 篇文檔；"
              f"source_pack_id={self.index.get('source_pack_id')}")

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """BM25-lite 檢索。回傳索引層資訊（不含全文）。"""
        q = tokenize(query)
        if not q:
            return []
        n = len(self.docs)
        avg_title = sum(len(t) for t in self._title_tok) / max(n, 1)
        avg_head = sum(len(t) for t in self._head_tok) / max(n, 1)

        scored: List[Tuple[float, int]] = []
        for i, d in enumerate(self.docs):
            tf, df = self._title_tok[i], self._head_tok[i]
            score = 0.0
            for term in q:
                if term in d["title"].lower():
                    score += 3.0
                if term in tf:
                    score += 1.0
                if term in df:
                    score += 0.6
                if term in d["primary_category"].lower():
                    score += 1.5
            if score > 0:
                # 輕微 IDF 懲罰：出現在越多文件中的詞，權重越低
                dfreq = sum(1 for t in self._head_tok if term in t)
                idf = 1.0 + (n / (1.0 + dfreq)) ** 0.5
                scored.append((score * idf, i))
        scored.sort(key=lambda x: -x[0])

        results = []
        for score, i in scored[:top_k]:
            d = self.docs[i]
            # 命中句：掃描前 30 句，回傳最多 3 個命中句的摘要片段（不給全文）
            hits = []
            for si, s in enumerate(d["sentences"][:30]):
                if any(t in s.lower() for t in q):
                    hits.append({"s_idx": si, "snippet": s[:220]})
                if len(hits) >= 3:
                    break
            results.append({
                "doc_id": d["doc_id"],
                "title": d["title"],
                "primary_category": d["primary_category"],
                "published": d["published"],
                "score": round(score, 3),
                "hit_sentences": hits,
            })
        return results

    def read(self, doc_id: str, max_sentences: int = 12) -> Optional[Dict[str, Any]]:
        """讀取文檔全文（帶全域句號），這是 citation 的回查錨點。

        回傳**前 max_sentences 句**（arXiv 論文的前段通常是摘要與引言，
        事實密度最高）。s_idx 為全域句號，citation 可直接引用。
        """
        d = self.by_id.get(doc_id)
        if d is None:
            return None
        window = d["sentences"][:max_sentences]
        return {
            "doc_id": d["doc_id"],
            "title": d["title"],
            "authors": d["authors"][:6],
            "primary_category": d["primary_category"],
            "published": d["published"],
            "url": d["url"],
            "n_sentences": d["n_sentences"],
            # 只回傳**適合引用**的句子，但保留其全局句號（citation 座標不變）
            "sentences": [
                {"s_idx": i, "text": t}
                for i, t in enumerate(window) if not is_noisy(t)
            ],
            "truncated": d["n_sentences"] > max_sentences,
        }


# ---------------------------------------------------------------------------
# 工具結果記錄
# ---------------------------------------------------------------------------
@dataclass
class ToolCall:
    step: int
    tool: str
    arguments: Dict[str, Any]
    observation: Any
    ok: bool
    error: str = ""
    latency_ms: int = 0


@dataclass
class ToolBox:
    """工具箱：封裝三個工具並記錄全部呼叫。

    第二輪新增：**候選 allowlist**。
    read_source 只能讀取「最近一次 search_sources 實際回傳過的 doc_id」，
    這是為了保證 Agent 的來源選擇確實來自檢索結果，而非憑記憶或僥倖命中。
    """
    pack: SourcePack
    calls: List[ToolCall] = field(default_factory=list)
    brief_text: Optional[str] = None
    brief_sources: List[str] = field(default_factory=list)
    # 候選 allowlist：每次 search_sources 成功後更新為該次回傳的 doc_id 集合
    last_candidates: List[str] = field(default_factory=list)
    all_candidates: List[str] = field(default_factory=list)
    # 任務要求的互異來源數；讀滿後工具不再允許讀取更多
    required_sources: int = 3

    # ---- 工具實作 ----
    def search_sources(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        hits = self.pack.search(query, top_k=top_k)
        ids = [h["doc_id"] for h in hits]
        # 更新 allowlist：最近一次檢索的候選集合
        self.last_candidates = ids
        for i in ids:
            if i not in self.all_candidates:
                self.all_candidates.append(i)
        return {
            "n_hits": len(hits),
            "results": hits,
            "note": ("索引層結果：僅含標題/類別與命中句片段，未含全文。"
                     "請對候選 doc_id 呼叫 read_source 取得全文後再引用。"),
        }

    def read_source(self, doc_id: str) -> Dict[str, Any]:
        # allowlist：只能讀「曾經由 search_sources 回傳過」的候選。
        # 注意：允許讀取**先前任一次**檢索的候選，而非僅最近一次——
        # 否則 Agent 在讀過某篇之後再檢索一次，就會失去讀該篇的權利，
        # 導致「已讀文件不可重讀」與「只能讀最近候選」兩個約束互相衝突。
        pool = self.all_candidates or self.last_candidates
        if pool and doc_id not in pool:
            return {
                "error": (
                    f"doc_id {doc_id} 不在任何一次 search_sources 的候選清單內。"
                    f"可讀取的候選為: {pool}"
                )
            }
        # 任務只需三個來源；讀夠之後不再允許讀取更多。
        # 這是工具層的約束：避免 Agent 把有限的步數預算花在過度檢索上，
        # 使它必須把預算轉用於撰寫簡報。模型仍自主決定「讀哪三篇」。
        already = self.called_docs()
        if len(already) >= self.required_sources:
            return {
                "error": (
                    f"你已經讀取 {len(already)} 篇來源"
                    f"（{already}），已滿足任務所需的三篇。"
                    f"不要再讀取其他文件，請立即呼叫 write_brief 撰寫簡報。"
                )
            }
        if doc_id in already:
            return {
                "error": (
                    f"doc_id {doc_id} 已經讀過了（已讀: {already}）。"
                    f"請改讀另一個尚未讀取的候選 doc_id，"
                    f"以確保讀到三篇不同的來源。可用候選: "
                    f"{[c for c in self.last_candidates if c not in already]}"
                )
            }
        doc = self.pack.read(doc_id)
        if doc is None:
            return {"error": f"doc_id 不存在：{doc_id}"}
        return {"document": doc}

    def write_brief(self, brief_text: str, cited_doc_ids: List[str]) -> Dict[str, Any]:
        self.brief_text = brief_text
        self.brief_sources = list(cited_doc_ids)
        return {
            "written": True,
            "chars": len(brief_text),
            "cited_doc_ids": list(cited_doc_ids),
        }

    # ---- 記錄 ----
    def record(self, call: ToolCall) -> None:
        self.calls.append(call)

    def called_docs(self) -> List[str]:
        """成功 read_source 過的互異 doc_id（依呼叫順序）。"""
        seen: List[str] = []
        for c in self.calls:
            if c.tool == "read_source" and c.ok:
                did = c.arguments.get("doc_id")
                if did and did not in seen:
                    seen.append(did)
        return seen
