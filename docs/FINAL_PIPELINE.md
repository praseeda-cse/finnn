# FinSight final pipeline: design

**Goal:** beat the best published TAT-DQA test result under the official scorer, **without training any model**,
using **open models only**, on **free** hardware. Plus an **automatic risk report** for any uploaded statement.

## 1. What we have to beat

| System | Training | Official TAT-DQA test EM / F1 |
|---|---|---|
| MHST (dataset baseline) | trained | 41.5 / 50.7 |
| Doc2SoarGraph | trained | 59.2 / 67.6 |
| TAT-LLM-7B / 13B / 70B | fine-tuned on TAT-DQA | 71.3 / 80.2 · 72.2 / 80.5 · **76.5 / 83.9** |
| Human | – | 84.1 / 90.8 |
| *Ours so far:* Qwen2-VL-2B zero-shot (Paper 1) | none | 8.7 / 17.4 |
| *Ours so far:* TAT-LLM-7B (4-bit), own arithmetic → **+ our executor** (Paper 2) | none by us | 53.2 / 61.3 → **69.0 / 77.1** |

Source: official leaderboard, https://nextplusplus.github.io/TAT-DQA/ (checked Oct 2026; no newer entries).

**Why beating 76.5 without training is realistic.** On TAT-QA (same family, text + tables) a 2025 study found
strong LLMs with *no training*, plus an external calculator or Python execution, reach **90–92% accuracy** on numerical
questions, against 73–84% with the model's own arithmetic (arXiv 2506.08726). Execution, which we already have, is the
biggest lever. The other gains come from a stronger open model, retrieved worked examples and an ensemble.

## 2. Constraints → design choices

| Constraint | Consequence |
|---|---|
| No training (no resources, did not work) | Everything is inference-time: retrieval, prompting, execution, voting, ensembling. Weights and thresholds are *chosen* on dev, never learned. |
| Open models only | Main model chosen from open models (Qwen, Gemma, Mistral, Phi…), served locally. Also supports the "runs offline, documents stay private" point from the user survey. |
| No money | Free Kaggle GPUs (2 x T4, 30 h/week) running models with **Ollama**. Free API tiers were checked and are too small: Groq free = gpt-oss-120b only, about 80 questions/day; Cerebras now needs a card. |
| TAT-DQA documents are only 1–3 pages | Page retrieval adds nothing, so RAG is used where it helps: **retrieving solved training examples**. Row retrieval (BM25) is used by the risk report and for long documents. |

## 3. The pipeline

```
PDF / page ─► READ   numbered rows with " | " column breaks (PDF text layer)                         data.py
          ─► RAG    4 most similar SOLVED training questions: their rows, gold program, unit         retrieve.py
          ─► LLM    best open model (chosen on dev) writes  evidence · type · program/span · unit     solve.py + llm.py
          ─► EXEC   Python executes the program (no eval, whitelisted syntax tree)                   program.py
          ─► CHECK  every operand / span must be on the page ("grounded")                           solve.py
          ─► VOTE   if the model and TAT-LLM-7B disagree: 4 more samples + weighted vote             solve.py
          ─► OUT    answer + unit + program + highlighted rows + confidence (vote share)
          ─► RISK   automatic risk questions → verified line items → ratios in Python → Low/Med/High  risk.py
```

### 3.1 Example RAG (retrieve.py): measured on dev, no GPU
Each question gets the 4 most similar *solved* training questions. The model sees the evidence rows, the unit rows,
the gold program and the scale of each. This teaches TAT-DQA's conventions, which the official scorer is strict about:
the sign of a change, ×100 for percentages, which unit an answer carries, and span vs. multi-span.
Similarity uses the question *template* (years and numbers masked), with a bonus for documents in the same money unit.
`scripts/retrieval_check.py` (all 1,644 dev questions, 4 examples each):

| Retrieval | Same answer type | Same unit | Same kind of calculation (arithmetic) |
|---|---|---|---|
| question words only (previous version) | 74.1% | 57.2% | 46.2% |
| **template + same-unit bonus (final)** | **76.8%** | **61.2%** | **53.0%** |

Only the training split is retrieved from. Dev and test answers are never shown to the model.

### 3.2 The model: picked by measurement, not by guess
Notebook 05 Part A runs several open models on the same 200 dev questions with the same prompt, examples and scorer.
The highest official EM wins. Candidates that fit 2 x T4 in 4-bit: `qwen2.5:32b`, `qwen3:32b`, `gemma3:27b`,
`mistral-small3.2:24b`, `phi4:14b`, plus any newer open model of that size on ollama.com/library.

### 3.3 Exact arithmetic and grounding
The model never writes a computed number: it writes a program, and `program.py` executes it. Check: gold answers
pushed through the full pipeline score **99.64% EM on dev**, which is the ceiling of the answer format. Everything
below that is the model's error, not ours. A candidate is *grounded* when all its operands and spans occur in the
document; **97.3% of gold answers are grounded**, so the check rarely penalises correct answers. Ungrounded
candidates get a lower vote weight.

### 3.4 Ensemble: specialist + generalist, voting only where needed
TAT-LLM-7B is fine-tuned on TAT-DQA by its authors, so it knows the dataset's conventions. The general model reasons
better. Their errors should differ. When the two agree, the answer is final. When they disagree (Part C measures how often),
the main model is sampled 4 more times and a weighted vote decides. The weights for
TAT-LLM and for ungrounded candidates are chosen on dev. Voting only on disagreements keeps the test run inside
Kaggle's free hours. The winner's vote share is the confidence, used for "I'm not sure" answers.

### 3.5 Automatic risk report (risk.py)
When a statement is loaded, seven risk questions are asked automatically:

| Question | Metric | High risk | Medium |
|---|---|---|---|
| Can the company pay its short-term obligations? | current ratio | < 1.0 | 1.0–1.5 |
| Does it rely heavily on borrowed money? | debt-to-equity (or liabilities-to-equity) | > 2.0 or negative equity | 1.0–2.0 |
| Can operating profit cover interest? | interest coverage | < 1.5 | 1.5–3.0 |
| Is it profitable? | net margin | loss | < 5% |
| Are sales falling? | revenue change | < −10% | < 0% |
| Is profitability getting worse? | net margin change | drop > 5 points | any drop |
| Are profits backed by cash? | operating cash flow / net income | negative cash flow | < 0.8 |

1. One model call *reads* the line items for each year, copying numbers as printed and citing the row.
2. Every value is checked against its cited row (whole-number match); unsupported values are dropped. This was tested:
   a made-up number is rejected, and so is "39" when the row says "391".
3. Python computes the ratios and flags. Missing items are reported as "not in this document", never guessed.
4. A table scan with no model flags line items that moved ≥ 50% year on year. On a real dev page it found current
   debt jumping from 20 to 696 while long-term debt fell 95% (debt falling due within a year).

The thresholds are general rules of thumb, and the report says so.

**Evaluation:**
- **(a)** Pipeline accuracy on the **risk-related TAT-DQA questions** (debt, liabilities, equity, margins, interest,
  operating cash flow: 218 of 1,663 test questions), computed automatically.
- **(b)** Share of extracted values that pass the row check.
- **(c)** A small hand-checked set of report pages, e.g. Indian annual reports, which ties to the survey's lakh/crore request.

## 4. Evaluation protocol (what makes the claim valid)
1. Everything is chosen on the **200-question dev sample** (seed 0, same as notebooks 01–02): the model, k,
   direct vs. reason, and the ensemble weights.
2. The **test set is run once** with the frozen setting (notebook 05 Part E).
3. Scoring uses the **official TAT-DQA scorer** (`tatqa_metric.py`), which checks the unit, the same as the leaderboard.
4. Report per answer type, the risk subset, ablations (no RAG / RAG / + ensemble), the paired significance test
   from Paper 2, and the confidence–coverage table.

## 5. Honest risks
- **It may not beat 76.5.** Fallback claims, all still true and publishable:
  - the best *training-free* result on TAT-DQA
  - the best result with ≤ 32B open models on free hardware
  - the size of each component's gain
- **Contamination:** TAT-QA and TAT-DQA have been public since 2021/2022, so open models may have seen them during
  pre-training. We cannot rule this out; state it as a limitation. Our ablations still compare like with like.
- **4-bit models** (Ollama) may score a little below full precision.
- **Free GPU time:** a ~32B model needs about 4–6 h for the test set on 2 x T4. Plan the runs: dev first, test once.

## 6. Run order
1. Notebook 02 (already done) → upload `tat-llm-7b-fft_dev200.jsonl` and `tat-llm-7b-fft_test.jsonl` to Kaggle.
2. Notebook 05 Part A (pick the model, 2–3 candidates first) → Part B (ablation) → Part C (ensemble) → Part D (risk).
3. Freeze the setting → Part E (test, once) → send `runs.zip`.
4. Rewrite Paper 2 around the final pipeline; Paper 1 stays as the diagnostic study that motivated it.
