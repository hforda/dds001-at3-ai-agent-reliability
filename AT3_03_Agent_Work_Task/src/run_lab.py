"""
run_lab.py — AT3 ③ 主執行腳本

用法：
    python src/run_lab.py --check
    python src/run_lab.py --run --max-steps 6 --out-dir results/run_maxsteps6
    python src/run_lab.py --compare --max-steps-a 3 --max-steps-b 6

執行流程（每次 --run）：
  1. 載入本地來源包（離線）
  2. 載入本地模型（AT3 ② 已下載的 Qwen2.5-Coder-3B-Instruct；不訓練、不呼叫 API）
  3. 由模型驅動的工具調用循環，直到 write_brief 或步數用盡
  4. 寫出 final-brief.md / trajectory.jsonl / tool-results.jsonl / provenance.json
  5. 執行 final-state validator，寫出 validator.json 與 evaluation.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent import AgentRunner  # noqa: E402
from contract import RESEARCH_TASK, build_task_contract, write_contract_and_schema  # noqa: E402
from tools import SourcePack, ToolBox  # noqa: E402
from validate_final_state import Validator, print_summary  # noqa: E402

def _resolve_model_path() -> str:
    """解析本地模型權重路徑。

    優先序：
      1. 環境變數 REACT_MODEL_PATH（明確指定，推薦）
      2. 環境變數 REACT_MODEL_DIR（指向 models/ 的上層目錄）
      3. 專案內 ./models/Qwen2.5-Coder-3B-Instruct（自行放置權重時）
      4. 同級工作區中的 AT3 02 目錄（沿用既有已下載權重）
    若都找不到，回傳預設路徑字串，由呼叫端給出明確錯誤訊息。
    """
    env_path = os.environ.get("REACT_MODEL_PATH")
    if env_path:
        return env_path
    env_dir = os.environ.get("REACT_MODEL_DIR")
    if env_dir:
        return str(Path(env_dir) / "Qwen2.5-Coder-3B-Instruct")
    local = ROOT / "models" / "Qwen2.5-Coder-3B-Instruct"
    if local.exists():
        return str(local)
    sibling = ROOT.parent / "AT3_02_技术复现" / "models" / "Qwen2.5-Coder-3B-Instruct"
    if sibling.exists():
        return str(sibling)
    return str(local)


DEFAULT_MODEL = _resolve_model_path()


def do_check() -> int:
    """零依賴自檢：來源包、模型檔案、Python 版本。"""
    print(f"Python {platform.python_version()}")
    print(f"platform {platform.platform()}")
    pack = ROOT / "source_pack"
    ok_pack = (pack / "documents.jsonl").exists() and (pack / "index.json").exists()
    print(f"{'READY' if ok_pack else 'NOT READY'} source_pack: {pack}")
    if ok_pack:
        idx = json.loads((pack / "index.json").read_text(encoding="utf-8"))
        print(f"  source_pack_id={idx.get('source_pack_id')} "
              f"n_documents={idx.get('n_documents')} "
              f"license={idx.get('dataset',{}).get('license')}")
    mp = Path(DEFAULT_MODEL)
    ok_model = (mp / "config.json").exists()
    print(f"{'READY' if ok_model else 'NOT READY'} model: {mp}")
    try:
        import torch
        import transformers
        print(f"  torch {torch.__version__} cuda_avail={torch.cuda.is_available()} "
              f"transformers {transformers.__version__}")
        if torch.cuda.is_available():
            print(f"  gpu {torch.cuda.get_device_name(0)} "
                  f"mem={torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    except ImportError as e:
        print(f"  torch/transformers 不可用：{e}")
    print(f"{'READY' if ok_pack and ok_model else 'NOT READY'} contract/schema")
    return 0


def _load_llm(model_path: str, dtype: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    td = {"bfloat16": torch.bfloat16, "float16": torch.float16,
          "float32": torch.float32}[dtype]
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=td, device_map=device if device == "cuda" else None)
    if device != "cuda":
        model = model.to(device)
    model.eval()
    model.generation_config.pad_token_id = tok.pad_token_id

    # 預熱：首次推理會觸發 CUDA kernel 編譯與 autotune（實測單次可達 270s），
    # 之後的推理僅數秒。預熱後再開始計時，避免把一次性開銷算進實驗耗時。
    import time as _t
    _t0 = _t.time()
    _enc = tok(["warmup text " * 64], return_tensors="pt", add_special_tokens=False)
    _enc = {k: v.to(model.device) for k, v in _enc.items()}
    with torch.inference_mode():
        model.generate(**_enc, max_new_tokens=8, do_sample=False,
                       pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    print(f"[load] GPU 預熱完成（{_t.time() - _t0:.1f}s）")

    class _LLM:
        """最小生成封裝（僅推理）。"""

        def __init__(self):
            self.tok = tok
            self.model = model

        def generate_chat(self, messages, stop_sequences=None, max_new_tokens=900):
            """以標準多輪對話結構生成（使用模型自帶的 chat template）。

            實測理由：手工拼接的純文字 prompt 在長上下文下會誘發 3B 模型
            **模仿對話歷史**（續寫 "Action: ... Observation: ..." 而非產出新規劃）。
            使用 chat template 後，每輪的 role 邊界明確，模型會正確地
            只產出 assistant 回合的內容。這不改變 Agent 的決策邏輯，
            只是讓提示結構符合模型的訓練分布。
            """
            text = self.tok.apply_chat_template(
                list(messages), tokenize=False, add_generation_prompt=True)
            return self.generate([text], stop_sequences=stop_sequences,
                                 max_new_tokens=max_new_tokens)

        def generate(self, prompts, stop_sequences=None, max_new_tokens=900):
            enc = self.tok(list(prompts), return_tensors="pt", padding=True,
                           add_special_tokens=False)
            enc = {k: v.to(self.model.device) for k, v in enc.items()}
            stops = list(stop_sequences[0]) if stop_sequences else []
            with torch.inference_mode():
                out = self.model.generate(**enc, max_new_tokens=max_new_tokens,
                                         pad_token_id=self.tok.pad_token_id,
                                         eos_token_id=self.tok.eos_token_id,
                                         do_sample=False)
            in_len = enc["input_ids"].shape[1]
            texts = []
            for seq in out:
                t = self.tok.decode(seq[in_len:], skip_special_tokens=True)
                cut = len(t)
                for s in stops:
                    if s:
                        i = t.find(s)
                        if i != -1:
                            cut = min(cut, i)
                t = t[:cut]
                # stop 序列可能落在輸出開頭（例如模型直接換行接 Observation），
                # 此時切完會變成空字串 → 保留原始輸出，讓上層記錄為格式錯誤而非無輸出。
                if not t.strip():
                    t = self.tok.decode(seq[in_len:], skip_special_tokens=True)
                texts.append(t)
            return texts

    return _LLM()


def run_once(max_steps: int, out_dir: Path, model_path: str, dtype: str,
             device: str, seed: int, task: str) -> Dict[str, Any]:
    """執行一次完整實驗。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    import torch
    torch.manual_seed(seed)

    print(f"\n=== 實驗開始 max_steps={max_steps} -> {out_dir} ===")
    pack = SourcePack(ROOT / "source_pack")
    tb = ToolBox(pack=pack)

    print("[load] 模型 ...")
    llm = _load_llm(model_path, dtype, device)

    agent = AgentRunner(llm=llm, toolbox=tb, task=task, max_steps=max_steps)
    t0 = time.time()
    summary = agent.run()
    elapsed = time.time() - t0
    print(f"[run] 步數={summary['steps_used']}/{max_steps} "
          f"stop={summary['stop_reason']} 讀取來源={summary['sources_read']} "
          f"耗時={elapsed:.1f}s")

    # --- 寫出 final-brief.md ---
    brief = tb.brief_text or ""
    (out_dir / "final-brief.md").write_text(brief, encoding="utf-8")

    # --- 寫出 trajectory.jsonl 與 tool-results.jsonl ---
    # trajectory：Agent 的逐步決策紀錄（step / thought / action / observation 摘要 / 結果）
    # tool-results：每次工具呼叫的完整紀錄（含完整 observation）
    with (out_dir / "tool-results.jsonl").open("w", encoding="utf-8") as f:
        for c in tb.calls:
            f.write(json.dumps({
                "step": c.step,
                "tool": c.tool,
                "arguments": c.arguments,
                "ok": c.ok,
                "error": c.error,
                "latency_ms": c.latency_ms,
                "observation": c.observation,
            }, ensure_ascii=False) + "\n")

    # trajectory：Agent 的逐步決策紀錄（直接取自 agent.steps_log，
    # 與 tool-results 逐條對應，validator 會檢查兩者序列一致）
    with (out_dir / "trajectory.jsonl").open("w", encoding="utf-8") as f:
        for s in agent.steps_log:
            f.write(json.dumps({
                "step": s["step"],
                "thought": s.get("thought", ""),
                "action": {"tool": s["tool"], "arguments": s.get("arguments", {})},
                "observation_preview": s.get("observation_preview", ""),
                "ok": s["ok"],
                "error": s.get("error", ""),
                "raw_model_output": s.get("raw_model_output", ""),
            }, ensure_ascii=False) + "\n")

    # --- provenance.json ---
    idx = json.loads((ROOT / "source_pack" / "index.json").read_text(encoding="utf-8"))
    prov = {
        "mode": "local_llm_offline_agent",
        "task_id": build_task_contract()["task_id"],
        "source_pack_id": idx.get("source_pack_id"),
        "source_pack_path": "source_pack/",
        "source_pack_sha256": idx.get("documents_sha256"),
        "source_pack_raw_files": idx.get("dataset", {}).get("raw_files"),
        "source_dataset": {
            "name": idx.get("dataset", {}).get("name"),
            "source_url": idx.get("dataset", {}).get("source_url"),
            "subsets_used": idx.get("dataset", {}).get("subsets_used"),
            "license": idx.get("dataset", {}).get("license"),
            "downloaded_at": idx.get("dataset", {}).get("downloaded_at"),
        },
        "api_used": False,
        "model_used": True,
        "model_note": "本地開源權重推理（Qwen2.5-Coder-3B-Instruct）；非付費 API、非外部服務。",
        "network_used": False,
        "gpu_used": True,
        "gpu_note": "僅推理；無訓練、無微調。",
        "framework": "python-stdlib+transformers (no agent framework)",
        "agent_model": "Qwen/Qwen2.5-Coder-3B-Instruct",
        "agent_model_path": model_path,
        "agent_model_trained_by_us": False,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "max_steps": max_steps,
        "seed": seed,
        "run_started_at": started,
        "run_finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "wall_clock_sec": round(elapsed, 1),
        "source_pack_built_by_student": True,
        "source_pack_disclaimer": (
            "此來源包由學生依課程指示自行從 Hugging Face 公開資料集 "
            "(RealTimeData/arxiv_alltime) 下載並建構，**非教師發放的 fixture**。"
        ),
        "results_are_not_model_benchmarks": (
            "本實驗結果為單一研究任務下的 Agent 行為紀錄，不代表任何模型的通用能力。"
        ),
    }
    (out_dir / "provenance.json").write_text(
        json.dumps(prov, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- validator ---
    v = Validator(out_dir, ROOT / "source_pack", max_steps).run()
    print_summary(v)
    (out_dir / "validator.json").write_text(
        json.dumps(v, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- evaluation.csv ---
    with (out_dir / "evaluation.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for key in ("overall", "checks_passed", "checks_total", "max_steps",
                    "tool_calls", "distinct_citations", "distinct_sources_read"):
            w.writerow([key, v[key]])
        w.writerow(["steps_used", summary["steps_used"]])
        w.writerow(["stop_reason", summary["stop_reason"]])
        w.writerow(["format_errors", summary["format_errors"]])
        w.writerow(["wall_clock_sec", round(elapsed, 1)])
        for r in v["results"]:
            w.writerow([f"gate:{r['check']}", f"{r['status']} — {r['detail']}"])
        for c in tb.calls:
            w.writerow([f"step{c.step}:{c.tool}",
                        f"ok={c.ok} args={json.dumps(c.arguments, ensure_ascii=False)}"])
    return {"summary": summary, "validator": v, "agent": summary,
            "elapsed": elapsed, "out_dir": str(out_dir)}


def do_compare(a: int, b: int, model_path: str, dtype: str, device: str,
               seed: int, task: str) -> int:
    """受控比較：唯一變數為 max_steps。"""
    res_a = run_once(a, ROOT / "results" / f"run_maxsteps{a}", model_path, dtype,
                     device, seed, task)
    res_b = run_once(b, ROOT / "results" / f"run_maxsteps{b}", model_path, dtype,
                     device, seed, task)
    rows = [
        ("max_steps", a, b),
        ("validator_overall", res_a["validator"]["overall"],
         res_b["validator"]["overall"]),
        ("checks_passed", res_a["validator"]["checks_passed"],
         res_b["validator"]["checks_passed"]),
        ("tool_calls", res_a["validator"]["tool_calls"], res_b["validator"]["tool_calls"]),
        ("steps_used", res_a["agent"]["steps_used"], res_b["agent"]["steps_used"]),
        ("distinct_citations", res_a["validator"]["distinct_citations"],
         res_b["validator"]["distinct_citations"]),
        ("distinct_sources_read", res_a["validator"]["distinct_sources_read"],
         res_b["validator"]["distinct_sources_read"]),
        ("brief_written", res_a["agent"]["brief_written"], res_b["agent"]["brief_written"]),
        ("stop_reason", res_a["agent"]["stop_reason"], res_b["agent"]["stop_reason"]),
        ("format_errors", res_a["agent"]["format_errors"], res_b["agent"]["format_errors"]),
        ("wall_clock_sec", round(res_a["elapsed"], 1), round(res_b["elapsed"], 1)),
    ]
    with (ROOT / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", f"max_steps={a}", f"max_steps={b}"])
        w.writerows(rows)
    # 逐 gate 對照
    with (ROOT / "comparison_gates.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["gate", f"max_steps={a}", f"max_steps={b}", "changed_with_max_steps"])
        for ra, rb in zip(res_a["validator"]["results"], res_b["validator"]["results"]):
            w.writerow([ra["check"], ra["status"], rb["status"],
                        "yes" if ra["status"] != rb["status"] else "no"])
    print("\n=== 受控比較完成，結果見 comparison.csv / comparison_gates.csv ===")
    for name, va, vb in rows:
        print(f"  {name:24s} max_steps={a}: {va}   max_steps={b}: {vb}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--max-steps", type=int, default=6)
    ap.add_argument("--max-steps-a", type=int, default=3)
    ap.add_argument("--max-steps-b", type=int, default=6)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--model-path", default=DEFAULT_MODEL)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda" if os.environ.get("REACT_FORCE_CPU") != "1" else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    if a.check:
        return do_check()
    if a.compare:
        return do_compare(a.max_steps_a, a.max_steps_b, a.model_path, a.dtype,
                           a.device, a.seed, RESEARCH_TASK)
    if a.run:
        write_contract_and_schema()
        out = Path(a.out_dir) if a.out_dir else ROOT / "results" / f"run_maxsteps{a.max_steps}"
        run_once(a.max_steps, out, a.model_path, a.dtype, a.device, a.seed, RESEARCH_TASK)
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
