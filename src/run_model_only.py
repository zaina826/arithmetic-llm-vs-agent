"""
Run a *single* base model (no agents) on GSM8K-style questions and write JSONL outputs.

Key fixes:
- Robustly extract the model text from whatever aixplain returns (dict / object / string).
- Always store a clean one-line integer answer in `model_answer` (trim + regex extract).
- Keep the prompt exactly as you provided (normalized for everyone).
- Preserve fairness: same prompt prefix, one call per question, deterministic-ish settings if supported.
"""

import json
import re
from pathlib import Path
from typing import Any, Optional

from aixplain.factories import ModelFactory

QUESTIONS_PATH = "data/gsm8k_20.jsonl"
OUTPUT_PATH = "outputs/model_only_answers.jsonl"
MODEL_ID = "669a63646eb56306647e1091" 

PROMPT_PREFIX = (
    "IMPORTANT: Output ONLY the final numeric answer (an integer). "
    "No words, no units, no punctuation, no explanation, no extra lines."
)


_INT_RE = re.compile(r"-?\d+")


def _safe_get(d: Any, *keys: str) -> Optional[Any]:
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def extract_text(resp: Any) -> str:
   
    if resp is None:
        return ""

    if isinstance(resp, str):
        return resp

    if isinstance(resp, dict):
        v = _safe_get(resp, "data", "output")
        if isinstance(v, str):
            return v

        v = resp.get("output")
        if isinstance(v, str):
            return v

        v = resp.get("data")
        if isinstance(v, str):
            return v

        v = _safe_get(resp, "data", "data")
        if isinstance(v, str):
            return v

        v = _safe_get(resp, "choices", 0, "message", "content")  
        if isinstance(v, str):
            return v

        return json.dumps(resp, ensure_ascii=False)

    for attr in ("data", "output", "content", "text"):
        if hasattr(resp, attr):
            v = getattr(resp, attr)
            if isinstance(v, str):
                return v
            if isinstance(v, dict):
                # sometimes resp.data is dict
                vv = _safe_get(v, "output")
                if isinstance(vv, str):
                    return vv
                return json.dumps(v, ensure_ascii=False)

    return str(resp)


def normalize_integer_answer(text: str) -> str:
    s = (text or "").strip()
    matches = _INT_RE.findall(s)
    if not matches:
        return s
    return matches[-1]

def main():
    Path("outputs").mkdir(exist_ok=True)

    model = ModelFactory.get(MODEL_ID)

    with open(QUESTIONS_PATH, "r", encoding="utf-8") as fin, open(
        OUTPUT_PATH, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            row = json.loads(line)
            question = row["question"]

            prompt = PROMPT_PREFIX + question

            resp = model.run(prompt, max_tokens=1000)

            raw_text = extract_text(resp)
            answer = normalize_integer_answer(raw_text)

            fout.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "question": question,
                        "gold_answer": row.get("answer", row.get("gold_answer")),
                        "model_answer": answer,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print("Wrote model-only answers to", OUTPUT_PATH)


if __name__ == "__main__":
    main()
