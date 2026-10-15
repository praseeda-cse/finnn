# FinSight: grounded question answering on financial documents (TAT-DQA)

```
page (PDF / image) + question
   │
1. SEE      pages → numbered rows with word boxes (PDF text layer; OCR for scanned pages)
   │
2. REASON   TAT-LLM-7B (authors' pre-trained model, no training here) → type · evidence · equation · scale
   │
3. COMPUTE  safe Python executor runs the equation (x100 reconciliation for percentages)
   │
4. VERIFY   self-consistency voting · evidence grounded to page rows · highlighted on the page
   │
answer + unit + equation + highlighted evidence, graded by the official TAT-DQA scorer
```

No RAG (documents are 1–3 pages), no fine-tuning or reinforcement learning on our side: TAT-LLM
(Zhu et al., 2024; `next-tat/tat-llm-7b-fft`) is already fine-tuned on TAT-DQA by its authors.
Its prompt is rebuilt exactly as in the authors' released files (identical on all 1,644 dev questions).

## Layout
| Path | What |
|---|---|
| `finsight/data.py` | load packed data; render a page as numbered rows with ` \| ` column breaks |
| `finsight/tatllm.py` | Stage 2–4: TAT-LLM prompt, 5-step parser, executor + % reconciliation, sampling, voting, evidence grounding |
| `finsight/program.py` | safe expression evaluator (syntax-tree whitelist, no `eval`) |
| `finsight/evaluate.py` | official scorer + per-type EM/F1 + evidence recall/precision |
| `finsight/tatqa_metric.py`, `tatqa_utils.py` | **official** TAT-QA/TAT-DQA scorer (MIT, vendored) |
| `finsight/analysis.py` | error breakdown: which single fix (unit, sign, x100, rounding) would make each wrong answer correct |
| `finsight/demo.py` | read any digital PDF, answer, highlight evidence rows on the page |
| `finsight/prompt.py`, `infer.py` | general-LLM baseline (notebook 01) |
| `scripts/build_compact.py` | pack the TAT-DQA download → `data/*.json.gz` (19 MB) |
| `scripts/sanity_check.py` | gold answers through our format + executor + scorer = 99.7% EM on test |
| `scripts/make_notebooks.py` | generates the notebooks |
| `notebooks/01_baseline_general_llm.ipynb` | Qwen2.5-7B, no training (comparison row) |
| `notebooks/02_finsight_pipeline.ipynb` | **main results**: TAT-LLM vs. + executor vs. + voting; final test run |
| `notebooks/03_demo_app.ipynb` | Gradio demo |

## Run on Colab
- Baseline: https://colab.research.google.com/github/praseeda-cse/finnn/blob/main/notebooks/01_baseline_general_llm.ipynb
- Main pipeline: https://colab.research.google.com/github/praseeda-cse/finnn/blob/main/notebooks/02_finsight_pipeline.ipynb
- Demo: https://colab.research.google.com/github/praseeda-cse/finnn/blob/main/notebooks/03_demo_app.ipynb

**Runtime → Change runtime type → T4 GPU**, then run the cells. The first cell clones this repo and mounts
Google Drive; results go to **My Drive/finsight/runs/** and every step resumes after a disconnect.

## Reporting rules
- Tune on **dev**; run **test** once at the end (notebook 02, Part C).
- Report the official scorer's EM/F1 and the per-type table. TAT-LLM's published 72.6 EM comes from the
  authors' own lenient script (≈1% tolerance, ignores the unit); compare models under the same official scorer.

## Data
TAT-DQA © Zhu et al., CC BY 4.0 (https://github.com/NExTplusplus/TAT-DQA). TAT-LLM © its authors (Llama 2 licence).
