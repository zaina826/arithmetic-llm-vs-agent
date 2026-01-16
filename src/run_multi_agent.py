
import json
import re
from pathlib import Path
from typing import Any

from aixplain.factories import AgentFactory, TeamAgentFactory

QUESTIONS_PATH = "data/gsm8k_20.jsonl"
OUTPUT_PATH = "outputs/multi_agent_answers.jsonl"
MODEL_ID = "669a63646eb56306647e1091" 

PROMPT_PREFIX = (
    "IMPORTANT: Output ONLY the final numeric answer (an integer). "
    "No words, no units, no punctuation, no explanation, no extra lines."
)

_INT_RE = re.compile(r"-?\d+")


def extract_text(resp: Any) -> str:
    """Extract output text from common aiXplain response shapes."""
    if resp is None:
        return ""

    if isinstance(resp, dict):
        out = (
            resp.get("data", {}).get("output")
            or resp.get("output")
            or resp.get("data")
        )
        return out if isinstance(out, str) else str(resp)

    if hasattr(resp, "data"):
        data = getattr(resp, "data")
        if hasattr(data, "output"):
            out = getattr(data, "output")
            if isinstance(out, str):
                return out
        if isinstance(data, dict) and isinstance(data.get("output"), str):
            return data["output"]

    return str(resp)


def normalize_integer_answer(text: str) -> str:
    """Return the last integer found; if none, return stripped text for debugging."""
    s = (text or "").strip()
    nums = _INT_RE.findall(s)
    return nums[-1] if nums else s


def main():
    Path("outputs").mkdir(exist_ok=True)

    py_tool = AgentFactory.create_python_interpreter_tool()

    answerer = AgentFactory.create(
        name="Answerer",
        description="Solves the math word problem and proposes an integer answer.",
        instructions=(
            "Solve the math word problem.\n"
            "You may use the Python interpreter tool to compute.\n"
            "Return ONLY the final numeric answer as an integer."
        ),
        llm_id=MODEL_ID,
        tools=[py_tool],
    )

    checker = AgentFactory.create(
        name="Checker",
        description="Verifies whether the proposed integer answer is correct.",
        instructions=(
            "Verify the proposed answer for the given math word problem.\n"
            "You may use the Python interpreter tool to check calculations.\n"
            "If the answer is correct, reply with: OK\n"
            "If incorrect, reply with: WRONG: <short reason> and include the corrected integer."
        ),
        llm_id=MODEL_ID,
        tools=[py_tool],
    )

    corrector = AgentFactory.create(
        name="Corrector",
        description="Produces the final corrected integer answer if needed.",
        instructions=(
            "You will receive the question, an initial answer, and checker feedback.\n"
            "If the initial answer is wrong, compute the correct answer.\n"
            "You may use the Python interpreter tool.\n"
            "Return ONLY the final numeric answer as an integer."
        ),
        llm_id=MODEL_ID,
        tools=[py_tool],
    )

    team = TeamAgentFactory.create(
        name="VerifierTeam",
        agents=[answerer, checker, corrector],
        llm_id=MODEL_ID,
    )

    with open(QUESTIONS_PATH, "r", encoding="utf-8") as fin, open(
        OUTPUT_PATH, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            row = json.loads(line)
            question = row["question"]

            prompt = PROMPT_PREFIX + question

            resp = team.run(prompt)

            raw_text = extract_text(resp)
            answer = normalize_integer_answer(raw_text)

            fout.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "question": question,
                        "gold_answer": row["answer"],
                        "multi_agent_answer": answer,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print("Wrote multi-agent answers to", OUTPUT_PATH)


if __name__ == "__main__":
    main()
