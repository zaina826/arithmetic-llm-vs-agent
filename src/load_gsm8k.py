
import json
from datasets import load_dataset
from pathlib import Path

OUT_PATH = "data/gsm8k_20.jsonl"

def main():
    ds = load_dataset("gsm8k", "main", split="test")

    Path("data").mkdir(exist_ok=True)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for i in range(100):
            row = ds[i]
            f.write(json.dumps({
                "id": f"gsm{i:02d}",
                "question": row["question"],
                "answer": row["answer"].split("####")[-1].strip()
            }) + "\n")

    print("Saved 20 GSM8K problems to", OUT_PATH)

if __name__ == "__main__":
    main()
