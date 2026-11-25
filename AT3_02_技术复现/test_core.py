"""
test_core.py — 核心组件的单元测试（不依赖 GPU 与网络）

运行：
    python test_core.py

覆蓋：
  1. 答案归一化（照抄官方 wrappers.py 的規則，含論文中的標籤歧義案例）
  2. EM / F1 計算
  3. 動作解析器（含嵌套括號、JSON 風格、非法輸入）
  4. 四種 prompting 格式的 few-shot 構造與一致性（Act 與 ReAct 共用同一批軌跡）
  5. Wikipedia 環境的三動作語義（用 mock 替代網路）
  6. HotpotQA 抽樣與官方 seed=233 序列一致
  7. Wilson 置信區間
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

PASS, FAIL = 0, 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  {detail}")


def test_normalization() -> None:
    print("\n[1] 答案歸一化與指標")
    from src.evaluation import normalize_answer, exact_match, f1_score, compute_metrics
    # 官方 normalize_answer 的四步順序：lower -> punc -> articles -> whitespace
    check("去冠詞+標點+小寫", normalize_answer("The Quick, Brown Fox!") == "quick brown fox",
          repr(normalize_answer("The Quick, Brown Fox!")))
    check("多空白折叠", normalize_answer("a  b\tc") == "b c", repr(normalize_answer("a  b\tc")))
    check("數字帶逗號", normalize_answer("1,800 to 7,000 ft") == "1800 to 7000 ft")
    # 論文中的實際案例
    check("yes/Yes 相等", exact_match("yes", "Yes"))
    check("Saimaa 大小寫", exact_match("The Saimaa Gesture", "saimaa gesture"))
    check("多詞答案標點", exact_match("director, screenwriter, actor", "director screenwriter actor"))
    check("Israel/Israeli 不等（論文標籤歧義案例）", not exact_match("Israel", "Israeli"))
    f1, p, r = f1_score("the quick brown fox", "the quick red fox")
    check("F1=0.75", abs(f1 - 0.75) < 1e-9, f"{f1}")
    m = compute_metrics("Richard Nixon", "richard nixon")
    check("EM=1, F1=1", m["em"] == 1.0 and m["f1"] == 1.0)


def test_action_parser() -> None:
    print("\n[2] 動作解析器")
    from src.agent import parse_action, extract_finish_answer, extract_answer_line
    cases = [
        ("Thought 1: x\nAction 1: Search[Colorado orogeny]", ("search", "Colorado orogeny")),
        ("Action 2: Finish[Richard Nixon]", ("finish", "Richard Nixon")),
        ("Action 1: Lookup[named after]", ("lookup", "named after")),
        ("Action 1: search[bar]", ("search", "bar")),
        ("Action 1: search(foo)", ("search", "foo")),
        ("Action 1: Search[High Plains (United States)]", ("search", "High Plains (United States)")),
        ("Action 1: Search[Adam Clayton Powell (film)]", ("search", "Adam Clayton Powell (film)")),
        ('{"action": "Search[Adam Clayton Powell]"}', ("search", "Adam Clayton Powell")),
        ("garbage", ("", "")),
    ]
    for text, exp in cases:
        got = parse_action(text)
        check(f"parse {text[:40]!r}", got == exp, f"got={got} exp={exp}")
    check("finish 取方括號內容",
          extract_finish_answer("Action 3: Finish[1,800 to 7,000 ft]") == "1,800 to 7,000 ft")
    check("Answer 行解析",
          extract_answer_line("Thought: x\nAnswer: Richard Nixon") == "Richard Nixon")


def test_prompts() -> None:
    print("\n[3] 四種 prompting 格式")
    from src import prompts as P
    fs = {m: P.build_few_shot(m) for m in ("standard", "cot", "act", "react")}
    for m, blk in fs.items():
        check(f"{m} 非空", len(blk) > 100)
        check(f"{m} 含 6 個示例", blk.count("Question:") == 6, f"count={blk.count('Question:')}")
    # Act 與 ReAct 必須來自同一批動作/觀測（嚴格受控消融）
    act_actions = [a for a, _ in P.ACT_TRAJECTORIES[0]]
    react_block = fs["react"]
    for a in act_actions:
        check(f"ReAct 含 Act 的動作 {a}", a in react_block)
    check("Act 不含 Thought", "Thought" not in fs["act"])
    check("ReAct 含 Thought", "Thought 1:" in fs["react"])
    check("CoT 含 Let's think step by step", "Let's think step by step." in fs["cot"])
    check("Standard 只有 Q/A", "Action" not in fs["standard"] and "Thought" not in fs["standard"])
    # 論文附錄 C.1 的第一條示例逐字核對
    check("附錄 C.1 首題正確",
          "What is the elevation range for the area that the eastern sector" in fs["standard"])
    check("附錄 C.1 首答正確", "1,800 to 7,000 ft" in fs["standard"])
    check("ReAct 首條 thought 逐字",
          "I need to search Colorado orogeny, find the area that the eastern sector" in fs["react"])


def test_environment(monkeypatched: bool = True) -> None:
    print("\n[4] Wikipedia 環境（三動作語義，mock 網路）")
    from src.wikipedia_env import WikipediaEnv
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        env = WikipediaEnv(cache_dir=tmp)

        # mock：避免真實網路請求。_cached_get 內部呼叫 self._fetch，
        # 因此替換 _fetch 即可攔截所有網路存取。
        page_text = ("The Colorado orogeny was an episode of mountain building. "
                     "It is recorded in the Colorado orogen. "
                     "The belt extends south into New Mexico. "
                     "The orogeny was likely part of Yavapai orogeny. "
                     "More text here for a fifth sentence.")

        def fake_fetch(url):
            if "/search/page" in url:
                return {"pages": [{"title": "Colorado orogeny", "excerpt": "..."}]}
            if "Colorado" in url:
                return {"source": page_text}
            return None

        env._fetch = fake_fetch

        obs = env.step("search[Colorado orogeny]")
        check("search 返回前 5 句",
              obs.text.count(".") == 5 and "Colorado orogeny" in obs.text,
              f"sentences={obs.text.count('.')} text={obs.text[:90]}")
        check("search 命中則不返回 Could not find",
              not obs.text.startswith("Could not find"))

        obs2 = env.step("lookup[orogeny]")
        check("lookup 帶 (Result i / n) 前綴", obs2.text.startswith("(Result "), obs2.text[:40])
        obs3 = env.step("lookup[orogeny]")
        check("lookup 同 keyword 遞增游標", obs3.text.startswith("(Result 2 /"), obs3.text[:40])

        obs4 = env.step("finish[some answer]")
        check("finish 設定 done", obs4.done and obs4.reward == 1.0)
        check("finish 攜帶答案", obs4.info.get("answer") == "some answer")

        obs5 = env.step("think[reasoning here]")
        check("think 不改變環境且非終止",
              obs5.text == "Nice thought." and not obs5.done)

        obs6 = env.step("bogus[x]")
        check("非法動作回報", obs6.text == "Invalid action: bogus[x]")

        # 回合狀態重置
        env.reset_episode()
        obs7 = env.step("lookup[orogeny]")
        check("reset_episode 後 lookup 失效（無當前頁）", obs7.text == "No more results.\n",
              obs7.text)


def test_sampling() -> None:
    print("\n[5] 抽樣與統計")
    import pandas as pd
    from src.data import sample_subset
    from src.evaluation import wilson_interval
    df = pd.DataFrame({"id": [str(i) for i in range(7405)]})
    s1 = sample_subset(df, 100, seed=233)
    s2 = sample_subset(df, 100, seed=233)
    check("抽樣可重現", s1["id"].tolist() == s2["id"].tolist())
    # 與官方抽樣序列一致
    idxs = list(range(7405))
    random.Random(233).shuffle(idxs)
    official = sorted(idxs[:100])
    check("與官方 seed=233 序列一致", sorted(int(x) for x in s1["id"]) == official)
    check("抽樣數量正確", len(s1) == 100)
    lo, hi = wilson_interval(50, 100)
    check("Wilson 區間包含點估計", lo < 0.5 < hi, f"[{lo:.3f},{hi:.3f}]")
    lo0, hi0 = wilson_interval(0, 100)
    check("Wilson 區間在 0 成功時下界為 0", abs(lo0) < 1e-9 and hi0 > 0, f"[{lo0:.4f},{hi0:.4f}]")
    lo1, hi1 = wilson_interval(100, 100)
    check("Wilson 區間在 100% 成功時上界為 1", hi1 > 0.999, f"[{lo1:.4f},{hi1:.4f}]")


def test_batdc_vs_react_shared() -> None:
    print("\n[6] Act 與 ReAct 的受控消融（同一批人工軌跡）")
    from src import prompts as P
    from src.prompts import build_few_shot
    act_block = build_few_shot("act")
    react_block = build_few_shot("react")
    # 逐題比對：第 i 題的 Action/Observation 序列必須完全相同
    same = True
    for i in range(6):
        for action, obs in P.ACT_TRAJECTORIES[i]:
            if action not in act_block or action not in react_block:
                same = False
        if obs is not None:
            # observation 前缀（避免長文本截斷差異）必須在兩個塊中同時出現
            prefix = obs[:40]
            if (prefix in act_block) != (prefix in react_block):
                same = False
    check("兩格式共用同一批 Action/Observation", same)
    check("ReAct 塊比 Act 塊長（含 thought）", len(react_block) > len(act_block) * 1.3,
          f"{len(react_block)} vs {len(act_block)}")


def main() -> int:
    print("=" * 70)
    print("AT3 ② ReAct 复现 —— 核心组件单元测试")
    print("=" * 70)
    test_normalization()
    test_action_parser()
    test_prompts()
    test_environment()
    test_sampling()
    test_batdc_vs_react_shared()
    print("\n" + "=" * 70)
    print(f"结果：{PASS} 通过 / {FAIL} 失败")
    print("=" * 70)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
