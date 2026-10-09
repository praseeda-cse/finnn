# FinSight: grounded program QA on TAT-DQA

Step 1 of the FinSight-R1 plan: the **FIND → program → calculate → verify** core, built on the
text layer that TAT-DQA provides for every page. The vision parts (cell graph, zoom, RL) come later
and plug into this same evaluation.

```
page blocks (text + word boxes)  ──►  "[7] Cost of revenue | 43 % | 50 % | 55 %"   (column breaks from word gaps)
            + question
                 │
                 ▼
          LLM (Qwen2.5, QLoRA-fine-tuned)
                 │   evidence: 6
                 │   type: arithmetic
                 │   program: (55 + 50) / 2
                 │   scale: percent
                 ▼
     safe Python executor  ──►  52.5 (percent)  ──►  official TAT-DQA scorer (EM / F1, scale-aware)
```

The model **never does arithmetic**. It cites block ids (checked against gold `block_mapping`) and
writes a program. Training targets come from the gold `derivation` automatically: 98.9% of training
arithmetic questions execute exactly to the gold answer.

## Layout
| Path | What |
|---|---|
| `finsight/data.py` | load packed data, render a page as numbered blocks with ` \| ` column breaks |
| `finsight/program.py` | safe expression evaluator; gold derivation → training program |
| `finsight/prompt.py` | system prompt, training target, output parser |
| `finsight/infer.py` | 4-bit model loading, batched generation with resume |
| `finsight/train.py` | QLoRA SFT with resume from checkpoints |
| `finsight/evaluate.py` | official scorer + per-type EM/F1 + evidence recall/precision |
| `finsight/verify.py` | Step 5.4: checks answers against the cited evidence (operands, text, scale, format) |
| `finsight/trust.py` | Steps 5.6–5.7: verification-weighted voting, confidence, risk–coverage, abstention threshold |
| `finsight/reinforce.py` | Steps 4.2/4.5: evidence-aware reward and expert-iteration target selection |
| `finsight/demo.py` | Step 5.8: read any digital PDF, answer, highlight evidence on the page |
| `finsight/tatqa_metric.py`, `tatqa_utils.py` | **official** TAT-QA scorer (MIT, vendored) |
| `scripts/build_compact.py` | pack `fp/dataset` → `data/*.json.gz` (19 MB total) |
| `scripts/sanity_check.py` | oracle check: gold through our pipeline = **99.7% EM on test** |
| `scripts/smoke_test.py` | tiny train + predict on a laptop GPU |
| `scripts/make_bundle.py`, `make_notebooks.py` | build the Drive upload zip / the notebooks |
| `notebooks/01_zero_shot_baseline.ipynb` | no training, 7B few-shot |
| `notebooks/02_finetune_and_eval.ipynb` | QLoRA fine-tune, then dev and test evaluation |
| `notebooks/03_trust_and_reinforce.ipynb` | voting + confidence + abstention; expert iteration |
| `notebooks/04_demo_app.ipynb` | Gradio demo: upload PDF, ask, see answer + highlighted evidence |
| `scripts/test_trust.py` | CPU-only tests of voting, reward and expert-iteration selection |

## Run on Colab (browser)
Open a notebook straight from GitHub:
- Baseline: https://colab.research.google.com/github/praseeda-cse/finnn/blob/main/notebooks/01_zero_shot_baseline.ipynb
- Fine-tune: https://colab.research.google.com/github/praseeda-cse/finnn/blob/main/notebooks/02_finetune_and_eval.ipynb
- Trust + RL: https://colab.research.google.com/github/praseeda-cse/finnn/blob/main/notebooks/03_trust_and_reinforce.ipynb
- Demo: https://colab.research.google.com/github/praseeda-cse/finnn/blob/main/notebooks/04_demo_app.ipynb

Then: **Runtime → Change runtime type → T4 GPU → Save**, then **Runtime → Run all**. The first cell clones this repo
and asks to mount Google Drive. Results go to **My Drive/finsight/runs/**, so they survive disconnects.
After pushing code changes to GitHub, just re-run the first cell.

## Run locally (RTX 4050, 6 GB)
Fine for inference with ≤3B models in 4-bit and for small experiments. Use Colab for 7B models and full training.
Open the notebooks with your local Python kernel; they detect that they're not on Colab.

## Reporting rules (so numbers survive a panel)
- Tune on **dev**. Run **test** once at the end.
- Always report the official scorer's EM/F1 and the per-type table.
- The previous Qwen2-VL results (16.2% EM) used a non-official metric without scale. Re-score them with `finsight.evaluate.score` before comparing.
