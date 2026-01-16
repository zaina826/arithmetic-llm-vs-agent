import json
import re

FILES = {
    "model_only": ("outputs/model_only_answers.jsonl", "model_answer"),
    "single_agent": ("outputs/single_agent_answers.jsonl", "agent_answer"),
    "multi_agent": ("outputs/multi_agent_answers.jsonl", "multi_agent_answer"),
}

NUM_RE = re.compile(r"-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?")

def normalize_to_int_str(x):
  
    if x is None:
        return None

    s = str(x).strip()
    if not s:
        return None

    nums = NUM_RE.findall(s)
    if not nums:
        return None

    last = nums[-1].replace(",", "")  

    if "." in last:
        try:
            f = float(last)
        except ValueError:
            return None
        if abs(f - round(f)) < 1e-9:
            return str(int(round(f)))
        return str(f).rstrip("0").rstrip(".")
    else:
        try:
            return str(int(last))
        except ValueError:
            return None

def main():
    for system, (path, key) in FILES.items():
        correct = 0
        total = 0
        missing_key = 0
        no_pred_num = 0

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)

                gold_raw = row.get("gold_answer")
                gold = normalize_to_int_str(gold_raw)

                if key not in row:
                    missing_key += 1
                    continue

                pred_raw = row.get(key)
                pred = normalize_to_int_str(pred_raw)

                total += 1
                if pred is None:
                    no_pred_num += 1
                    continue

                if gold is not None and pred == gold:
                    correct += 1

        acc = correct / total if total else 0
        print(f"{system}: {correct}/{total} | accuracy = {acc:.2f}")
        if missing_key:
            print(f"  (warning) {missing_key} rows missing key '{key}'")
        if no_pred_num:
            print(f"  (info) {no_pred_num} rows had no extractable number")

if __name__ == "__main__":
    main()
