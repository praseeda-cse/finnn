# FinSight: grounded question answering on financial documents (TAT-DQA)

**Final pipeline** (no training, open models only, free GPUs; design and evidence in [docs/FINAL_PIPELINE.md](docs/FINAL_PIPELINE.md)):
```
page (PDF) + question
   │
1. READ     numbered rows with " | " column breaks (PDF text layer)
2. RAG      4 most similar solved TRAINING questions (rows, gold program, unit) as worked examples
3. REASON   best open LLM (chosen on dev) → evidence rows · type · program or span · unit
4. COMPUTE  safe Python executor runs the program
5. VERIFY   operands/spans must be on the page · where it disagrees with TAT-LLM-7B: 4 samples + weighted vote
   │
answer + unit + program + highlighted rows + confidence, graded by the official TAT-DQA scorer
   +
AUTOMATIC RISK REPORT: liquidity, leverage, interest cover, profitability, growth, cash backing,
large year-on-year changes; values checked against their rows, ratios computed in Python
```
Target: best published TAT-DQA test result, TAT-LLM-70B 76.5 EM / 83.9 F1 (official leaderboard).

The earlier pipeline (TAT-LLM-7B + executor + voting, notebook 02) gives 69.0 EM on test and is now one member of the ensemble.

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
| `finsight/solve.py` | **final pipeline**: prompt with retrieved examples, parsing, execution, grounding, weighted vote / ensemble |
| `finsight/retrieve.py` | **RAG**: example retrieval from the training split (question template + unit) and BM25 row retrieval |
| `finsight/llm.py` | client for any OpenAI-compatible server (Ollama, vLLM, llama.cpp); parallel, resumable |
| `finsight/risk.py` | **automatic risk report**: verified line items → ratios → Low/Medium/High; large-change scan |
| `finsight/prompt.py`, `infer.py` | answer format and parser; general-LLM baseline (notebook 01) |
| `finsight/vlm.py` | Qwen2-VL-2B on page images (paper 1 re-run, notebook 04) |
| `scripts/build_compact.py` | pack the TAT-DQA download → `data/*.json.gz` (19 MB) |
| `scripts/sanity_check.py` | gold answers through our format + executor + scorer = 99.7% EM on test |
| `scripts/make_notebooks.py` | generates the notebooks |
| `scripts/retrieval_check.py` | how often retrieved examples match the dev question's type, unit and calculation (CPU) |
| `scripts/analyse_user_study.py` | survey analysis: trust tests, groups, feature requests and charts from the form export |
| `notebooks/01_baseline_general_llm.ipynb` | Qwen2.5-7B, no training (comparison row) |
| `notebooks/02_finsight_pipeline.ipynb` | **main results**: TAT-LLM vs. + executor vs. + voting; final test run |
| `notebooks/03_demo_app.ipynb` | Gradio demo |
| `notebooks/04_paper1_vlm_rerun.ipynb` | Qwen2-VL-2B controlled re-run (budget, prompt, structured output) |
| `notebooks/05_final_pipeline.ipynb` | **final pipeline** on Kaggle: model choice, ablation, ensemble, risk, test run, demo app |
| `docs/FINAL_PIPELINE.md` | design of the final pipeline, with the evidence for each part |
| `user_study/` | user-study plan, questionnaire and the script that builds the Google Form |

## Installation

**Option A: Google Colab (recommended, needs no setup).** Use the links below. Notebooks install
their own packages. TAT-LLM-7B needs a GPU, and a free T4 is enough.

**Option B: local machine.** Needs Python 3.10+ and git. A CUDA GPU with ≥16 GB is needed only to run the models.
```bash
git clone https://github.com/praseeda-cse/finnn.git
cd finnn
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
python scripts/sanity_check.py  # CPU only, ~10 s: should print ~99.7% EM on test
```
The packed dataset is already in `data/`, so no separate download is needed. Model weights
(`next-tat/tat-llm-7b-fft`, `Qwen/Qwen2-VL-2B-Instruct`) download from Hugging Face on first use.

## Run on Colab
- Baseline: https://colab.research.google.com/github/praseeda-cse/finnn/blob/main/notebooks/01_baseline_general_llm.ipynb
- Main pipeline: https://colab.research.google.com/github/praseeda-cse/finnn/blob/main/notebooks/02_finsight_pipeline.ipynb
- Demo: https://colab.research.google.com/github/praseeda-cse/finnn/blob/main/notebooks/03_demo_app.ipynb

**Runtime → Change runtime type → T4 GPU**, then run the cells. The first cell clones this repo and mounts
Google Drive; results go to **My Drive/finsight/runs/** and every step resumes after a disconnect.

## Run the final pipeline on Kaggle (free, 2 x T4)
1. kaggle.com → Create → New notebook → File → Import notebook → GitHub `praseeda-cse/finnn`, `notebooks/05_final_pipeline.ipynb`.
2. Right panel: Accelerator **GPU T4 x2**, Internet **on**, Persistence **Files only**.
3. Add data → Upload the TAT-LLM outputs from notebook 02 (`tat-llm-7b-fft_dev200.jsonl`, `tat-llm-7b-fft_test.jsonl`).
4. Run Parts A–D on dev, then set `RUN_TEST = True` for Part E once. Download `runs.zip` at the end.

## Reporting rules
- Tune on **dev**; run **test** once at the end (notebook 05, Part E).
- Report the official scorer's EM/F1 and the per-type table. TAT-LLM's published 72.6 EM comes from the
  authors' own lenient script (≈1% tolerance, ignores the unit); compare models under the same official scorer.

## Data and licence
Code: MIT (see `LICENSE`). TAT-DQA © Zhu et al., CC BY 4.0 (https://github.com/NExTplusplus/TAT-DQA).
TAT-LLM © its authors (Llama 2 licence). Version history is in `CHANGELOG.md`.

## Team
Final-year project, Dept. of CSE, Amrita School of Computing, Coimbatore.
Vijaya Sree Praseeda · Neha T · Mahasri M · K. Shivani Shreya Reddy · Nivasini J. Guide: Dr. Bagyammal T.

Papers (drafts): *Reading Is Not Reasoning* (diagnostic study of a zero-shot VLM on TAT-DQA) ·
*FinSight* (this pipeline) · *Question Answering over Financial Documents: A Survey*.
