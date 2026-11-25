"""
contract.py — AT3 ③ 的任務契約與工具 schema

本檔案的 task-contract.json 與 tool-schema.json 由這裡產生，確保：
  1. 契約與 schema 在兩組受控比較實驗之間**完全一致**（唯一變數是 max_steps）；
  2. schema 描述的是**真實可呼叫**的工具，參數名稱與 tools.py 實作逐一對應。

契約內容對應課程 §③ 的要求：
  - 搜尋指定來源包
  - 讀取三個相關來源
  - 生成一頁含 citation 的研究簡報
  - 通過 final-state validator
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parent.parent

TASK_ID = "at3-03-llm-eval-literature-brief"

# ---------------------------------------------------------------------------
# 研究任務本體（兩組實驗共用，逐字相同）
# ---------------------------------------------------------------------------
RESEARCH_TASK = (
    "整理來源包中關於『大型語言模型（LLM）評估方法』的相關文獻，"
    "回答：近年的 LLM 評估方法在評估對象（評估什麼）與評估可靠性"
    "（如何確保評分可信）兩個維度上，主要有哪些技術路線與代表性工作？"
    "請從至少三篇彼此相關的來源中提取證據，"
    "並在簡報中以 citation 標註每一項事實的來源（doc_id + 標題 + 句號）。"
)

SUCCESS_CRITERIA = [
    "在來源包中執行 search_sources 檢索並取得候選來源",
    "對三個**彼此不同**的 doc_id 執行 read_source 並取得實際閱讀的正文"
    "（doc_id 必須來自最近一次 search_sources 的候選清單）",
    "生成 final-brief.md，內含至少三個互異 citation，每個 citation 可回溯到 (doc_id, s_idx)",
    "每個 citation 所指的來源句原文，必須逐字出現在帶該 marker 的那一條 bullet 內"
    "（僅容許空白字符歸一化）",
    "final-state validator 的全部檢查項通過",
]

CONSTRAINTS = [
    "全程離線：執行階段不得訪問網路",
    "不得呼叫付費 API 或外部模型服務",
    "不得使用 GPU 以外的加速裝置；不訓練、不微調任何模型",
    "僅使用來源包中的文檔作為證據，不得憑記憶補充事實",
    "citation 必須能逐字回查至來源包原文",
    "工具呼叫次數不得超過本次設定的 max_steps（每次工具呼叫計為一個 step）",
]

# ---------------------------------------------------------------------------
# 工具 schema（與 tools.py 逐一對應）
# ---------------------------------------------------------------------------
TOOL_SCHEMA: Dict[str, Any] = {
    "schema_version": "1.0",
    "source_pack_id": "at3-03-arxiv-2023q1",
    "tools": [
        {
            "name": "search_sources",
            "description": (
                "在本地離線來源包中檢索相關文檔。回傳的是索引層結果："
                "每個候選的 doc_id、標題、arXiv 主類別、命中句片段。"
                "**不回傳全文**——要取得可引用的正文，必須再對 doc_id 呼叫 read_source。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "檢索查詢字串，以研究問題或關鍵概念描述你要找的內容。",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回的候選數量，預設 5。",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
            "returns": {
                "n_hits": "int，命中候選數",
                "results": "list，每項含 doc_id / title / primary_category / published / score / hit_sentences",
            },
            "side_effects": "none（唯讀）",
            "counts_as_step": True,
        },
        {
            "name": "read_source",
            "description": (
                "讀取來源包中的一篇文檔全文，回傳帶全域句號的句子列表。"
                "**這是 citation 的唯一合法證據來源**：簡報中的每一項事實都必須"
                "能在此回傳的 (doc_id, s_idx) 處逐字找到。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_id": {
                        "type": "string",
                        "description": ("要讀取的文檔 ID，必須來自最近一次 search_sources 回傳的候選清單，"
                                       "格式如 NNNN.NNNNNvN（例如 <doc_id_from_search_results>）。"),
                    },
                },
                "required": ["doc_id"],
            },
            "returns": {
                "document": "dict，含 doc_id / title / authors / primary_category / published / url / sentences[]",
            },
            "side_effects": "none（唯讀）",
            "counts_as_step": True,
        },
        {
            "name": "write_brief",
            "description": (
                "生成最終研究簡報並寫入 final-brief.md。"
                "brief 內的每個事實後面必須標註 citation 標記，格式為 "
                "[doc_id:<doc_id>#sN]，其中 N 是該句在 read_source 回傳的句子列表中的編號。"
                "只允許引用你**確實 read_source 過**的 doc_id。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "brief_text": {
                        "type": "string",
                        "description": "完整的一頁簡報 Markdown 文字（含標題、摘要、主題分類與 citation 標記）。",
                    },
                    "cited_doc_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "簡報中實際引用到的 doc_id 列表，須為已讀取來源的子集。",
                    },
                },
                "required": ["brief_text", "cited_doc_ids"],
            },
            "returns": {
                "written": "bool",
                "chars": "int，簡報字元數",
                "cited_doc_ids": "list",
            },
            "side_effects": "write final-brief.md",
            "counts_as_step": True,
        },
    ],
}


def build_task_contract() -> Dict[str, Any]:
    """產生 task-contract.json 的內容（不含實驗特定參數）。"""
    return {
        "contract_version": "1.0",
        "task_id": TASK_ID,
        "task": RESEARCH_TASK,
        "goal": "在離線來源包中完成一次來源整理工作，產出一頁可追溯引用的研究簡報。",
        "success_criteria": SUCCESS_CRITERIA,
        "constraints": CONSTRAINTS,
        "required_sources": {
            "count": 3,
            "must_be_distinct": True,
            "definition": "三次成功的 read_source 呼叫，且 doc_id 互異。",
        },
        "source_pack": {
            "id": "at3-03-arxiv-2023q1",
            "path": "source_pack/",
            "documents_file": "source_pack/documents.jsonl",
            "index_file": "source_pack/index.json",
            "offline": True,
        },
        "agent": {
            "kind": "local_llm_tool_loop",
            "model": "Qwen/Qwen2.5-Coder-3B-Instruct",
            "decides_next_action": True,
            "note": "由模型根據任務與工具回傳結果決定下一步操作，非預設固定順序。",
        },
        "tools": [t["name"] for t in TOOL_SCHEMA["tools"]],
        "step_accounting": {
            "definition": "每次工具呼叫計為一個 step。",
            "max_steps_is_the_compared_variable": True,
        },
        "offline_requirements": {
            "api_used": False,
            "model_used": "local_open_weights_only",
            "network_used": False,
            "gpu_used": "inference_only_no_training",
            "large_frameworks": False,
        },
    }


def write_contract_and_schema() -> None:
    """把兩個契約檔案寫到工程根目錄（交付物之一）。"""
    (ROOT / "task-contract.json").write_text(
        json.dumps(build_task_contract(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (ROOT / "tool-schema.json").write_text(
        json.dumps(TOOL_SCHEMA, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("已寫出 task-contract.json 與 tool-schema.json")


if __name__ == "__main__":
    write_contract_and_schema()
