# LLM Illusion of Explanatory Depth (IOED) Evaluation

Do large language models overestimate their own understanding — and does asking them to explain themselves reveal that gap?

This project tests whether LLMs exhibit an analogue of the **Illusion of Explanatory Depth (IOED)**, first documented in humans by Rozenblit & Keil (2002). In humans, self-rated understanding drops significantly after attempting a step-by-step mechanistic explanation. We investigate whether the same effect appears in LLMs, and whether it is domain-specific in the way it is for humans.

---

## Research Design

Each model is run through a fixed **three-turn protocol** per question:

| Turn | Prompt | What we capture |
|---|---|---|
| 1 | Brief answer + confidence rating (0–10) | `first_answer`, `first_rating` (R1) |
| 2 | Explain step-by-step how you arrived at your answer | `explanation` |
| 3 | How confident are you now? Rate 0–10 | `second_rating` (R2) |

The key signal is **Δ = R2 − R1**. A negative delta means the model became less confident after explaining — the IOED-like effect. Turn 2 deliberately does not ask the model to re-evaluate its answer, to avoid demand characteristics.

---

## Datasets

Three categories of questions, chosen to probe different regimes of human/AI competence:

| Dataset | File | Category | Expected IOED |
|---|---|---|---|
| Rozenblit devices & procedures | `data/rozenblit_dataset.json` | Hard for humans, accessible for AI | Small delta (AI can explain these) |
| Easy Problems | `data/easyProblems.json` | Easy for humans, hard for AI | Large delta (AI overconfident) |
| SPARTUN | `data/spartun_100_samples.json` | Spatial reasoning — easy for humans, hard for AI | Large delta |
| HLE | `data/hle_test.jsonl` | Hard for both humans and AI | Moderate/small delta |

The **Rozenblit devices vs. procedures** comparison is the core replication test. Human IOED appears on devices (tacit mechanical knowledge) but not procedures (things people can actually articulate). If models mirror this, devices should show larger drops than procedures.

---

## Project Structure

```
explanatory-depth-project/
├── data/                          # All question datasets
│   ├── rozenblit_dataset.json
│   ├── easyProblems.json
│   ├── spartun_100_samples.json
│   └── hle_test.jsonl
│
├── evals/
│   ├── core/
│   │   └── pipeline.py            # Shared API, prompts, rating extraction, ETA
│   ├── rozenblit/
│   │   └── eval_rozenblit.py
│   ├── easy_problems/
│   │   └── eval_easy_problems.py
│   ├── spartun/
│   │   └── eval_spartun.py
│   └── hle/
│       └── eval_hle.py
│
├── results/                       # Output JSON files (timestamped per run)
├── llm_eval_list_v2.csv           # Model list with OpenRouter IDs
└── llm_pipeline.ipynb             # Original exploratory notebook (archived)
```

---

## Setup

**1. Install dependencies**
```bash
pip install requests python-dotenv
```

**2. Add your OpenRouter API key**

Create a `.env` file in the project root:
```
OPENROUTER_API_KEY=your_key_here
```

**3. Select models**

Open `llm_eval_list_v2.csv` and set `IOED_Include = Yes` for the models you want to run. Then update the `MODELS` list in `evals/core/pipeline.py` with their `OpenRouter_ID` values.

> Note: 10 models in the CSV have `OpenRouter_ID = N/A` and cannot be run through this pipeline (Claude 3 Opus, Gemini 1.5 Pro, Grok 2, Llama 3.1 405B, Jamba 1.5, Yi-1.5 34B, InternLM 2.5, EXAONE 3.5, Falcon 3 10B, Granite 3.2).

---

## Running Evals

Each eval is a standalone script. Run from the project root:

```bash
python evals/rozenblit/eval_rozenblit.py
python evals/easy_problems/eval_easy_problems.py
python evals/spartun/eval_spartun.py
python evals/hle/eval_hle.py
```

Results are saved incrementally to `results/` after each entry — safe to `Ctrl+C` and resume. Live output shows R1, R2, Δ, and a rolling ETA per question.

To run models in parallel, open separate terminal tabs and edit `MODELS` in `pipeline.py` to a single model per tab.

---

## Output Format

Each result file is a JSON array. Common fields across all evals:

```json
{
  "question_id": "device_01",
  "question": "How does a zipper work?",
  "category": "Devices",
  "model": "anthropic/claude-3-5-haiku",
  "first_answer": "A zipper works by interlocking teeth. 7",
  "first_rating": 7,
  "explanation": "Step 1: ...",
  "second_rating": 5,
  "error": null
}
```

SPARTUN results additionally include `first_selected`, `first_correct`, `second_selected`, `second_correct` for accuracy grading.

---

## Key Hypotheses

1. **Domain shift**: Human IOED is strongest on devices/mechanisms. LLM IOED should be strongest on spatial and novel reasoning tasks — where generating a coherent explanation is genuinely difficult — not on devices, where LLMs have extensive training coverage.

2. **Devices vs. procedures replication**: If the Rozenblit pattern holds for LLMs, devices should show larger confidence drops than procedures.

3. **Calibration baseline**: HLE questions serve as a baseline — models likely already have lower confidence here, leaving less room for IOED.

---

## References

- Rozenblit, L. & Keil, F. (2002). The misunderstood limits of folk science: an illusion of explanatory depth. *Cognitive Science*, 26(5), 521–562.
- Phan, L. et al. (2025). Humanity's Last Exam. arXiv:2501.14249.
- Mirzaee, R. et al. (2021). SPARTQA: A Textual Question Answering Benchmark for Spatial Reasoning. arXiv:2104.05832.
- Williams, S. & Huckle, J. (2024). Easy Problems That LLMs Get Wrong. arXiv:2405.19416.
