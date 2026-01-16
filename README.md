# Single vs Multi-Agent Arithmetic Reasoning

This project compares how different interaction structures affect LLM performance on arithmetic word problems:

- model-only (direct LLM call)
- single-agent (LLM + Python tool)
- multi-agent (Answerer–Checker–Corrector + Python tool)

All experiments use the same GSM8K subset and are evaluated with exact-match accuracy.

---

## Structure

```text
singlevsmulti/
├── data/gsm8k_20.jsonl
├── outputs/
│   ├── model_only_answers.jsonl
│   ├── single_agent_answers.jsonl
│   └── multi_agent_answers.jsonl
├── src/
│   ├── load_gsm8k.py
│   ├── run_model_only.py
│   ├── single_agent.py
│   ├── run_multi_agent.py
│   └── evaluate_math.py
````

---

## Setup

```bash
pip install aixplain numpy pandas tqdm
export AIXPLAIN_API_KEY="YOUR_API_KEY"
```

---

## Run Experiments

```bash
python src/run_model_only.py
python src/single_agent.py
python src/run_multi_agent.py
```

Outputs are saved in `outputs/`.

---

## Evaluate

```bash
python src/evaluate_math.py
```

---

## Results

```text
model_only:   0.42
single_agent: 0.91
multi_agent:  0.76
```

---

## Additional Exploratory Notebook

The notebook `arithmetic-llm.ipynb` contains exploratory analysis conducted to better understand why transformer-based language models struggle with arithmetic. It uses GPT-2 as a case study due to its open weights and inspectability.

This notebook is **not part of the experimental pipeline**, is **not used to produce any reported results**, and does **not affect the evaluation**. It was created solely as supplementary analysis to deepen understanding of numerical representation, embeddings, attention, and arithmetic failure modes in LLMs.


## Takeaway

Accuracy improves mainly due to **Python tool access**, not agent multiplicity.
Single-agent systems provide the best balance for arithmetic tasks.


## AI Usage Disclosure

I used AI assistance only to improve the README and documentation.  
All code, experiments, and results were written and implemented by me.
