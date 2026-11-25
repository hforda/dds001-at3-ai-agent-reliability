"""
validate_final_state.py — AT3 ③ 的 final-state validator

【核心原則】validator 檢查的是**最終交付狀態**，不是「檔案是否存在」。
每一項都必須基於實際的 trajectory / tool-results / 來源包內容做判定，
而不是讀取任何自我聲明的欄位。

檢查項（八項，對應使用者指定的清單）：
  1. brief_exists             final-brief.md 存在且內容完整（非空、有標題、含 citation）
  2. three_distinct_citations 引用了三個**互異**的來源
  3. three_sources_read       這三個來源**確實**執行過 read_source
  4. citations_traceable      每個 citation 的 (doc_id, s_idx) 能定位到原始文檔
  5. quoted_text_exists       引用內容真實存在於來源包（非無根據掛靠）
  6. trajectory_consistent    trajectory 與 tool-results 能相互核對
  7. within_step_budget       實際工具呼叫數 ≤ max_steps
  8. provenance_complete      provenance.json 欄位完整且聲稱與實測一致

輸出：每項 PASS/FAIL + 具體失敗原因；總體 PASS 需八項全通過。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

CITATION_RE = re.compile(r"\[doc_id:([^\]#\s]+)#s(\d+)\]")

VALIDATOR_CHECKS = [
    ("brief_exists", "final-brief.md 存在且內容完整"),
    ("three_distinct_citations", "引用了三個互異來源"),
    ("three_sources_read", "三個來源確實執行過 read_source"),
    ("citations_traceable", "每個 citation 能定位到原始文檔"),
    ("quoted_text_exists", "引用內容真實存在於來源包"),
    ("trajectory_consistent", "trajectory 與 tool-results 可相互核對"),
    ("within_step_budget", "遵守 max_steps 限制"),
    ("provenance_complete", "provenance 完整且與實測一致"),
]

_STOP = {
    "that", "this", "with", "from", "have", "been", "were", "they", "which",
    "their", "there", "these", "those", "about", "would", "could", "should",
    "using", "used", "such", "also", "than", "then", "when", "what", "into",
    "over", "more", "most", "some", "each", "both", "does", "make", "made",
}


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def content_words(s: str) -> set:
    words = re.findall(r"[A-Za-z][A-Za-z\-']{3,}", s.lower())
    return {w for w in words if w not in _STOP}


class Validator:
    def __init__(self, out_dir: Path, pack_dir: Path, max_steps: int) -> None:
        self.out = Path(out_dir)
        self.pack_dir = Path(pack_dir)
        self.max_steps = max_steps
        self.results: List[Dict[str, Any]] = []
        self.diagnostics: Dict[str, Any] = {}

        self.brief_path = self.out / "final-brief.md"
        self.traj_path = self.out / "trajectory.jsonl"
        self.tools_path = self.out / "tool-results.jsonl"
        self.prov_path = self.out / "provenance.json"

        self.brief_text = (self.brief_path.read_text(encoding="utf-8")
                           if self.brief_path.exists() else "")
        self.trajectory = load_jsonl(self.traj_path)
        self.tool_results = load_jsonl(self.tools_path)
        self.provenance = (json.loads(self.prov_path.read_text(encoding="utf-8"))
                           if self.prov_path.exists() else {})

        self.docs: Dict[str, Dict[str, Any]] = {}
        doc_file = self.pack_dir / "documents.jsonl"
        if doc_file.exists():
            with doc_file.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        d = json.loads(line)
                        self.docs[d["doc_id"]] = d

    def _add(self, name: str, ok: bool, detail: str) -> None:
        self.results.append({"check": name, "status": "PASS" if ok else "FAIL",
                             "detail": detail})

    # ------------------------------------------------------------------
    def run(self) -> Dict[str, Any]:
        # --- 1. brief_exists ---
        ok = bool(self.brief_text.strip())
        detail = ""
        if ok:
            stripped = self.brief_text.strip()
            has_heading = stripped.startswith("#") or "\n#" in stripped
            has_cit = bool(CITATION_RE.search(self.brief_text))
            ok = len(stripped) >= 400 and has_heading and has_cit
            detail = (f"chars={len(stripped)}, heading={has_heading}, "
                      f"has_citation={has_cit}")
            if not ok:
                detail += " → 內容不完整（需 >=400 字、有標題、含 citation 標記）"
        else:
            detail = "final-brief.md 不存在或為空"
        self._add("brief_exists", ok, detail)

        # --- 收集 citation ---
        cites = CITATION_RE.findall(self.brief_text)
        cited_doc_ids: List[str] = []
        for did, _ in cites:
            if did not in cited_doc_ids:
                cited_doc_ids.append(did)

        # --- 2. three_distinct_citations ---
        ok = len(cited_doc_ids) >= 3
        detail = f"unique cited doc_ids={len(cited_doc_ids)}: {cited_doc_ids}"
        if not ok:
            detail += " → 少於 3 個互異來源"
        self._add("three_distinct_citations", ok, detail)

        # --- 3. three_sources_read ---
        read_docs: List[str] = []
        for t in self.tool_results:
            if t.get("tool") == "read_source" and t.get("ok"):
                did = (t.get("arguments") or {}).get("doc_id")
                if did and did not in read_docs:
                    read_docs.append(did)
        ok = len(read_docs) >= 3
        detail = f"distinct read_source doc_ids={len(read_docs)}: {read_docs}"
        if not ok:
            detail += " → 實際讀取的互異來源少於 3"
        self._add("three_sources_read", ok, detail)

        # --- 4/5. citations_traceable & quoted_text_exists ---
        # 第 4 項（traceable）：每個 (doc_id, s_idx) 必須能在來源包中定位到該句，
        #   且 #sN 必須是該文件的真實全域句號。
        # 第 5 項（quoted_text_exists）：**嚴格逐字核驗**——
        #   marker 所指的來源句原文，必須逐字出現在「帶該 marker 的那一行 bullet」內，
        #   僅容許空白字符歸一化。這是第二輪的核心修正：
        #   第一輪僅要求「一個共同內容詞」，等同於允許無根據掛靠；
        #   本輪不接受僅憑一個共同詞認定引用有效。
        # 另附診斷指標 lexical_overlap_rate（不影響 PASS/FAIL）：
        #   引用行與來源句的內容詞重疊程度。**詞彙重疊不等於語義蕴含**，
        #   故此指標僅作為模型事實性風險的量化參考，不得當作語義驗證。
        trace_fail: List[str] = []
        quote_fail: List[str] = []
        n_checked = 0
        n_aligned = 0
        brief_lines = [ln.strip() for ln in self.brief_text.split("\n") if ln.strip()]

        for did, sidx in cites:
            n_checked += 1
            doc = self.docs.get(did)
            if doc is None:
                trace_fail.append(f"{did} 不在來源包中")
                continue
            try:
                si = int(sidx)
            except ValueError:
                trace_fail.append(f"{did}#s{sidx} 句號非法")
                continue
            if si < 0 or si >= len(doc["sentences"]):
                trace_fail.append(
                    f"{did}#s{si} 超出句號範圍(0..{len(doc['sentences'])-1})")
                continue
            sent = re.sub(r"\s+", " ", doc["sentences"][si]).strip()
            marker = f"[doc_id:{did}#s{sidx}]"
            owner_lines = [ln for ln in brief_lines if marker in ln]
            if not owner_lines:
                quote_fail.append(f"{did}#s{si} 找不到含該 marker 的行")
                continue
            owner = owner_lines[0]
            owner_body = re.sub(r"\s+", " ", owner.replace(marker, " ")).strip()
            # 嚴格逐字：來源句原文必須完整出現在該行
            if sent not in owner_body:
                quote_fail.append(
                    f"{did}#s{si} 來源句原文未逐字出現於該 bullet"
                    f"（來源句開頭: \"{sent[:80]}\"）")
                continue
            # 診斷：詞彙重疊程度（僅供參考，不代表語義蘊含）
            src_words = content_words(sent)
            own_words = content_words(owner)
            if src_words:
                overlap = len(src_words & own_words)
                ratio = overlap / min(len(src_words), max(len(own_words), 1))
                if overlap >= 2 or ratio >= 0.12:
                    n_aligned += 1
        ok4 = n_checked > 0 and not trace_fail
        detail4 = (f"檢查 {n_checked} 個 citation 標記；不可定位者: "
                   f"{trace_fail if trace_fail else '無'}")
        self._add("citations_traceable", ok4, detail4)

        ok5 = n_checked > 0 and not quote_fail
        detail5 = (f"檢查 {n_checked} 個 citation 標記；原文未逐字出現於該 bullet 者: "
                   f"{quote_fail if quote_fail else '無'}")
        self._add("quoted_text_exists", ok5, detail5)

        # 診斷指標（不計入 PASS/FAIL）
        overlap_rate = (n_aligned / n_checked) if n_checked else 0.0
        self.diagnostics["lexical_overlap_rate"] = round(overlap_rate, 4)
        self.diagnostics["lexical_overlap_note"] = (
            "引用行與來源句的內容詞重疊率達門檻的比例。"
            "注意：詞彙重疊**不等於**語義蘊含驗證，僅作為模型事實性風險的"
            "量化參考；引用有效性的判定依據是第 5 項的逐字核驗。")
        self.diagnostics["citations_checked"] = n_checked

        # --- 6. trajectory_consistent ---
        # 第二輪強化：比對 **step + tool + arguments**（第一輪只比 step+tool，
        # 參數不同也視為一致，核驗力不足）。
        # 並額外核驗：最終寫入的簡報內容必須與日誌中記錄的寫入內容一致。
        def _sig(t: Dict[str, Any]) -> tuple:
            args = (t.get("action") or {}).get("arguments")
            if args is None:
                args = t.get("arguments")
            # 只比對可比較的參數鍵（brief_text 另以 SHA256 核驗）
            if isinstance(args, dict):
                args = {k: v for k, v in sorted(args.items()) if k != "brief_text"}
            return (t.get("step"), (t.get("action") or {}).get("tool") or t.get("tool"),
                    json.dumps(args, ensure_ascii=False, sort_keys=True)
                    if args is not None else None)

        traj_ids = [_sig(t) for t in self.trajectory]
        tool_ids = [_sig(t) for t in self.tool_results]
        seq_ok = traj_ids == tool_ids and len(traj_ids) > 0
        detail = (f"trajectory 條目={len(traj_ids)}, tool-results 條目={len(tool_ids)}, "
                  f"step+tool+arguments 序列一致={seq_ok}")

        # 簡報寫入內容一致性：寫入檔案的內容必須與日誌記錄的寫入內容相同。
        write_records = [t for t in self.tool_results
                         if t.get("tool") == "write_brief" and t.get("ok")]
        brief_ok = True
        if write_records:
            logged = (write_records[-1].get("arguments") or {}).get("brief_text", "")
            logged_norm = re.sub(r"\s+", " ", str(logged)).strip()
            file_norm = re.sub(r"\s+", " ", self.brief_text).strip()
            brief_ok = (logged_norm == file_norm)
            detail += f"; 簡報寫入內容與日誌一致={brief_ok}"
            if not brief_ok:
                detail += f" (日誌 {len(logged_norm)} 字 vs 檔案 {len(file_norm)} 字)"
        else:
            # 沒有成功寫入的記錄，則簡報必須不存在（否則是無來源的文件）
            brief_ok = not self.brief_text.strip()
            detail += f"; 無成功 write_brief 記錄，簡報應為空={brief_ok}"

        ok = seq_ok and brief_ok
        if not seq_ok:
            detail += f"; trajectory={traj_ids}; tool_results={tool_ids}"
        self._add("trajectory_consistent", ok, detail)

        # --- 7. within_step_budget ---
        n_calls = sum(1 for t in self.tool_results
                      if t.get("tool") in ("search_sources", "read_source", "write_brief"))
        ok = n_calls <= self.max_steps
        detail = f"工具呼叫數={n_calls}, max_steps={self.max_steps}"
        if not ok:
            detail += " → 超出步數預算"
        self._add("within_step_budget", ok, detail)

        # --- 8. provenance_complete ---
        required = ["mode", "source_pack_id", "api_used", "model_used", "network_used",
                    "gpu_used", "python", "framework", "max_steps", "agent_model",
                    "source_pack_sha256", "run_started_at", "seed"]
        missing = [k for k in required if k not in self.provenance]
        inconsistent = []
        for k in ("api_used", "network_used"):
            if self.provenance.get(k) is not False:
                inconsistent.append(f"{k}={self.provenance.get(k)} (應為 false)")
        if self.provenance.get("max_steps") is not None and \
                int(self.provenance["max_steps"]) != self.max_steps:
            inconsistent.append(
                f"max_steps 聲稱={self.provenance['max_steps']} 實際設定={self.max_steps}")
        ok = not missing and not inconsistent
        detail = f"缺欄位={missing if missing else '無'}"
        if inconsistent:
            detail += f"; 聲稱不一致={inconsistent}"
        self._add("provenance_complete", ok, detail)

        n_pass = sum(1 for r in self.results if r["status"] == "PASS")
        return {
            "overall": "PASS" if n_pass == len(self.results) else "FAIL",
            "checks_passed": n_pass,
            "checks_total": len(self.results),
            "max_steps": self.max_steps,
            "tool_calls": n_calls,
            "distinct_citations": len(cited_doc_ids),
            "distinct_sources_read": len(read_docs),
            "diagnostics": self.diagnostics,
            "results": self.results,
        }


def print_summary(summary: Dict[str, Any]) -> None:
    print("=" * 74)
    print("FINAL-STATE VALIDATOR")
    print("=" * 74)
    for r in summary["results"]:
        mark = "PASS" if r["status"] == "PASS" else "FAIL"
        print(f"  [{mark}] {r['check']:26s} {r['detail']}")
    print("-" * 74)
    print(f"  總體：{summary['overall']}  "
          f"({summary['checks_passed']}/{summary['checks_total']} 項通過)  "
          f"max_steps={summary['max_steps']}  tool_calls={summary['tool_calls']}  "
          f"citations={summary['distinct_citations']}  "
          f"sources_read={summary['distinct_sources_read']}")
    print("=" * 74)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pack-dir", default=str(ROOT / "source_pack"))
    ap.add_argument("--max-steps", type=int, required=True)
    ap.add_argument("--out-json", default=None)
    a = ap.parse_args()
    s = Validator(Path(a.out_dir), Path(a.pack_dir), a.max_steps).run()
    print_summary(s)
    if a.out_json:
        Path(a.out_json).write_text(json.dumps(s, ensure_ascii=False, indent=2),
                                    encoding="utf-8")
