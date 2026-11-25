#!/usr/bin/env python
"""第二輪核心邏輯的單元測試（不需模型、不需 GPU）。

覆蓋：
  T1  read_source allowlist：非候選 doc_id 被拒
  T2  write_brief 嚴格逐字核驗：原句逐字貼上 → 通過
  T3  write_brief 嚴格逐字核驗：改寫（paraphrase）→ 拒絕
  T4  write_brief 嚴格逐字核驗：句號錯誤 → 拒絕
  T5  write_brief 嚴格逐字核驗：引用未讀過的文檔 → 拒絕
  T6  write_brief 嚴格逐字核驗：少於三個互異來源 → 拒絕
  T7  無 marker → 拒絕
  T8  重複 marker 於多行 → 拒絕
  T9  空白字符歸一化視為通過（換行/多空格）
  T10 validator 的 quoted_text_exists 與 lexical_overlap_rate 命名
  T11 trajectory_consistent 比對 arguments
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent import AgentRunner, parse_action  # noqa: E402
from tools import SourcePack, ToolBox  # noqa: E402

PASSED = 0
FAILED = 0


def check(name: str, cond: bool, extra: str = "") -> None:
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  [PASS] {name}")
    else:
        FAILED += 1
        print(f"  [FAIL] {name} {extra}")


def make_agent(pack_dir: Path):
    pack = SourcePack(pack_dir)
    tb = ToolBox(pack=pack)
    ag = AgentRunner(llm=None, toolbox=tb, task="test", max_steps=10)
    return pack, tb, ag


def main() -> int:
    pack_dir = ROOT / "source_pack"
    if not (pack_dir / "documents.jsonl").exists():
        print("[SKIP] source_pack 不存在，無法執行單元測試")
        return 0

    pack, tb, ag = make_agent(pack_dir)

    # 找兩篇真實文檔作為測試素材
    dids = [d["doc_id"] for d in pack.docs[:2]]
    d0, d1 = dids[0], dids[1]
    s0 = pack.by_id[d0]["sentences"][3]
    s1 = pack.by_id[d1]["sentences"][2]

    print("=" * 70)
    print("T1  read_source allowlist")
    print("=" * 70)
    tb.search_sources("evaluation", top_k=5)
    cand = tb.last_candidates
    check("T1.1 檢索後 last_candidates 非空", len(cand) > 0, f"cand={cand}")
    # 找一個不在候選中的 doc_id
    outside = next(d["doc_id"] for d in pack.docs if d["doc_id"] not in cand)
    res_bad = tb.read_source(outside)
    check("T1.2 非候選 doc_id 被 allowlist 拒絕",
          "error" in res_bad and "候選" in res_bad["error"], str(res_bad)[:120])
    res_ok = tb.read_source(cand[0])
    check("T1.3 候選 doc_id 可正常讀取", "document" in res_ok)

    print()
    print("=" * 70)
    print("T2-T9  write_brief 嚴格逐字核驗")
    print("=" * 70)

    # 準備已讀三篇（繞過 allowlist：直接注入 calls 記錄）
    d2 = pack.docs[2]["doc_id"]
    for did in (d0, d1, d2):
        tb.calls.append(type(tb.calls[0])(1, "read_source", {"doc_id": did},
                                         "ok", True, "", 0)) if tb.calls else None
    # 若 calls 為空則手動構造
    if not tb.calls:
        from tools import ToolCall
        for i, did in enumerate((d0, d1, d2)):
            tb.calls.append(ToolCall(i + 1, "read_source", {"doc_id": did},
                                      "obs", True, "", 1))
    else:
        from tools import ToolCall
        tb.calls = []
        for i, did in enumerate((d0, d1, d2)):
            tb.calls.append(ToolCall(i + 1, "read_source", {"doc_id": did},
                                      "obs", True, "", 1))

    # T2 逐字正確
    brief_ok = (f"# Title\n\nSummary line about the topic.\n\n"
                f"- {s0} [doc_id:{d0}#s3]\n"
                f"- {s1} [doc_id:{d1}#s2]\n"
                f"- {pack.by_id[d2]['sentences'][1]} [doc_id:{d2}#s1]")
    obs, ok, err = ag._invoke("write_brief", {"brief_text": brief_ok})
    check("T2 原文逐字貼上 → 接受", ok, f"err={err} obs={obs[:160]}")

    # T3 改寫 → 拒絕
    paraphrase = brief_ok.replace(s0, "This paper shows a method for the task.")
    obs, ok, err = ag._invoke("write_brief", {"brief_text": paraphrase})
    check("T3 改寫原句 → 拒絕", (not ok) and err == "citation_not_verbatim",
          f"ok={ok} err={err}")

    # T4 句號錯誤 → 拒絕（句號指向別的句子）
    wrong_idx = brief_ok.replace(f"#s3]", f"#s5]")
    obs, ok, err = ag._invoke("write_brief", {"brief_text": wrong_idx})
    check("T4 句號指向錯誤句子 → 拒絕",
          (not ok) and err == "citation_not_verbatim", f"ok={ok} err={err}")

    # T5 引用未讀文檔 → 拒絕
    unread = pack.docs[50]["doc_id"]
    brief_unread = brief_ok.replace(f"[doc_id:{d0}#s3]", f"[doc_id:{unread}#s3]")
    obs, ok, err = ag._invoke("write_brief", {"brief_text": brief_unread})
    check("T5 引用未讀過的文檔 → 拒絕", (not ok) and err == "cited_unread",
          f"ok={ok} err={err}")

    # T6 少於三個互異來源 → 拒絕
    # 註：第二輪把所有問題合併為一次回饋，故此例會以 citation_not_verbatim
    # 回報（訊息中同時包含「只引用了 2 個互異來源」的說明）。
    two = (f"# Title\n\nSummary.\n\n"
           f"- {s0} [doc_id:{d0}#s3]\n"
           f"- {s1} [doc_id:{d1}#s2]")
    obs, ok, err = ag._invoke("write_brief", {"brief_text": two})
    check("T6 少於三個互異來源 → 拒絕",
          (not ok) and "互異來源" in obs, f"ok={ok} err={err} obs={obs[:150]}")

    # T7 無 marker → 拒絕
    nomark = "# Title\n\nSummary.\n\n- A finding with no citation at all."
    obs, ok, err = ag._invoke("write_brief", {"brief_text": nomark})
    check("T7 無 citation marker → 拒絕",
          (not ok) and err == "no_citation_markers", f"ok={ok} err={err}")

    # T8 同一 marker 出現在多行 → 拒絕
    dup = brief_ok + f"\n- repeated {s0} [doc_id:{d0}#s3]"
    obs, ok, err = ag._invoke("write_brief", {"brief_text": dup})
    check("T8 同一 marker 出現在多行 → 拒絕",
          (not ok) and err == "citation_not_verbatim", f"ok={ok} err={err}")

    # T9 空白歸一化：把原句中的單空格換成多空格/換行，仍應通過
    spaced = s0.replace(" ", "  ")
    brief_sp = brief_ok.replace(s0, spaced)
    obs, ok, err = ag._invoke("write_brief", {"brief_text": brief_sp})
    check("T9 空白字符歸一化後仍視為逐字命中", ok, f"err={err} obs={obs[:150]}")

    print()
    print("=" * 70)
    print("T10-T11  validator 檢查")
    print("=" * 70)
    vsrc = (ROOT / "src" / "validate_final_state.py").read_text(encoding="utf-8")
    check("T10.1 已無 semantic_alignment_rate 命名",
          "semantic_alignment_rate" not in vsrc)
    check("T10.2 使用 lexical_overlap_rate 命名",
          "lexical_overlap_rate" in vsrc)
    check("T10.3 引用有效判定為逐字核驗（sent not in owner_body）",
          "sent not in owner_body" in vsrc)
    check("T10.4 明確聲明詞彙重疊不等於語義蘊含",
          "不等於" in vsrc and "語義蘊含" in vsrc)
    check("T11 trajectory_consistent 比對 arguments",
          "step+tool+arguments" in vsrc and "arguments" in vsrc)

    print()
    print("=" * 70)
    print(f"結果：{PASSED} 通過 / {FAILED} 失敗")
    print("=" * 70)
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
