# Changelog

Versions are git tags (`git tag`); each line links to the commits that made it.

## Unreleased: security layer
- `integrity.py`: numeric text layer vs. rendered page with OCR, hidden-text detection and removal.
- `gate.py`: binds every number to a box; Verified / Conflict / Abstain from rules, no model.
- `secure.py`: document access control and a hash-chained audit log around `solve.answer`.
- `scripts/tamper_eval.py` on dev (CPU, no model): results in `docs/FINAL_PIPELINE.md` section 3.6.
- Notebook 05 runs end to end with one Run All: Part E on by default (setting still frozen on dev first),
  2 candidate models, results zipped before the demo starts, setup reports missing TAT-LLM files.
- `solve.span_scale`: a copied span that already states its unit ("$2.7 million", "12%") or is plain text
  gets no separate scale (97–100% of such training answers have none); `-0` is written as 0 and a span
  copied twice is kept once. Found by error analysis on dev (72.5 → 75.5 EM), then applied once to the
  saved test outputs with no new model calls: test 66.1 → 68.4 EM / 75.9 → 77.8 F1.

## v0.5 (2026-10-25): final pipeline
- Example RAG over the training split (question template + same-unit bonus, tuned on dev with
  `scripts/retrieval_check.py`): retrieved examples match the question's calculation 53.0% vs 46.2% before.
- `solve.py`: any open LLM via `llm.py` (Ollama / vLLM / llama.cpp), executor, grounding check, weighted
  ensemble with TAT-LLM-7B and voting only where they disagree; gold answers through it score 99.64% EM on dev.
- `risk.py`: automatic risk report (7 risk questions, row-verified values, ratios in Python, large-change scan).
- Notebook 05 for Kaggle (2 x T4, free): model choice, ablation, ensemble, risk, single test run, demo app.
- Design document `docs/FINAL_PIPELINE.md`.

## v0.4 (2026-10-25)
- Installation guide, `requirements.txt`, MIT licence, this changelog.
- User study: use case and user groups, 19-question survey, Apps Script that builds the Google Form
  (`user_study/`), analysis script (`scripts/analyse_user_study.py`).

## v0.3 (2026-10-15)
- Final pipeline: TAT-LLM-7B + executor + self-consistency voting + evidence grounding.
- Error breakdown (`finsight/analysis.py`): which single fix would make each wrong answer correct.
- Notebook 04: Qwen2-VL-2B controlled re-run for paper 1 (longer budget, generic prompt, structured output + executor).
- Batches sized by token budget; out-of-memory retries; guarded test-set run in notebook 02.

## v0.2 (2026-10-12)
- TAT-LLM stage with the authors' prompt rebuilt exactly; percentage reconciliation in the executor.
- No-fine-tuning path with retrieved in-context examples.

## v0.1 (2026-10-04 – 2026-10-09)
- First grounded program-QA pipeline for TAT-DQA: data packing, safe executor, official scorer, sanity check.
- Verifier, voting / confidence / abstention, demo app.
