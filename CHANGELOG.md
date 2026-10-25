# Changelog

Versions are git tags (`git tag`); each line links to the commits that made it.

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
