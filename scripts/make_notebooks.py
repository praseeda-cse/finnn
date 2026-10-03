"""Generate the Colab notebooks in notebooks/. Edit the cell text here, then re-run.

    python scripts/make_notebooks.py
"""
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

SETUP = r'''# --- Setup: on Colab, gets the latest code + data from GitHub; also works on a local GPU ---
import os, sys, subprocess, shutil, random
IN_COLAB = os.path.exists("/content")
if IN_COLAB:
    ROOT = "/content/finnn"
    shutil.rmtree(ROOT, ignore_errors=True)               # always fetch the latest code
    subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/praseeda-cse/finnn.git", ROOT], check=True)
    try:                                                  # save results on Drive so they survive disconnects
        from google.colab import drive
        drive.mount("/content/drive")
        OUT = "/content/drive/MyDrive/finsight/runs"
    except Exception as e:
        print("Drive not mounted (", e, ") - results are saved in /content/runs and are LOST if Colab disconnects")
        OUT = "/content/runs"
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "peft", "bitsandbytes", "accelerate"], check=True)
else:
    ROOT = os.path.abspath("..")
    OUT = os.path.join(ROOT, "runs")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, ROOT)

import torch
from finsight.data import load_split
from finsight.evaluate import score, print_report
from finsight.infer import load_model, predict
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE - select a GPU runtime!")

docs_train, q_train = load_split(f"{ROOT}/data/tatdqa_train.json.gz")
docs_dev, q_dev = load_split(f"{ROOT}/data/tatdqa_dev.json.gz")
docs_test, q_test = load_split(f"{ROOT}/data/tatdqa_test.json.gz")
def sample(questions, n, seed=0):
    return random.Random(seed).sample(questions, n) if n < len(questions) else questions
print(len(q_train), len(q_dev), len(q_test), "questions")'''

SHOW_ERRORS = r'''# Look at mistakes: this is where the next improvement comes from
rows = {r["uid"]: r for r in result["rows"]}
wrong = [q for q in eval_qs if rows[q["uid"]]["em"] == 0]
for q in wrong[:10]:
    p = preds[q["uid"]]
    print(f"[{q['answer_type']}] {q['question']}")
    print(f"   gold: {q['answer']} ({q['scale']}) | evidence {q['evidence']} | derivation {q['derivation']!r}")
    print(f"   pred: {p['answer']} ({p['scale']}) | evidence {p['evidence']}")
    print("   raw:", p["raw"].replace(chr(10), " / "), "\n")'''


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text}


def notebook(cells):
    return {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"accelerator": "GPU", "kernelspec": {"name": "python3", "display_name": "Python 3"},
                         "language_info": {"name": "python"}}}


baseline = notebook([
    md("# 01 · Zero-shot baseline (no training)\n\n"
       "An instruction-tuned LLM reads the page text (with block ids and ` | ` column breaks), and writes "
       "`evidence / type / program-or-answer / scale`. Python executes the program, and the **official TAT-DQA scorer** grades it.\n\n"
       "Runtime: a T4 GPU takes about 15–25 min for 200 dev questions with the 7B model."),
    code(SETUP),
    code('MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"   # 4-bit, fits a 16 GB T4\n'
         'N = 200                                 # dev questions (random sample); use len(q_dev) for all 1,644\n'
         'eval_qs = sample(q_dev, N)\n'
         'name = MODEL_ID.split("/")[-1]\n\n'
         'model, tok = load_model(MODEL_ID)\n'
         'preds = predict(model, tok, docs_dev, eval_qs, f"{OUT}/zeroshot_{name}_dev{N}.jsonl", few_shot=True, batch_size=4)\n'
         'result = score(eval_qs, preds)\n'
         'print_report(result, f"{name} few-shot, dev sample")'),
    code(SHOW_ERRORS),
])

finetune = notebook([
    md("# 02 · Fine-tune (QLoRA) and evaluate\n\n"
       "Supervised fine-tuning on the 13k training questions. The targets are built automatically from the gold "
       "`derivation` and `block_mapping`: the model learns to cite evidence blocks and write a program, and never "
       "does arithmetic itself.\n\n"
       "**If Colab disconnects:** reconnect (Runtime → T4 GPU), then re-run the setup and training cells. Training resumes from the last checkpoint on Drive, "
       "and prediction resumes from its `.jsonl` file.\n\n"
       "Rough T4 timing for the 3B model: ~2–3 h for 1 epoch. For a quick first run, set `MAX_EXAMPLES = 3000` (~35 min)."),
    code(SETUP),
    code('from finsight.train import train\n\n'
         'MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"   # T4-friendly. With an A100/L4, try "Qwen/Qwen2.5-7B-Instruct"\n'
         'MAX_EXAMPLES = None                     # e.g. 3000 for a quick first run\n'
         'RUN = f"{OUT}/sft_{MODEL_ID.split(\'/\')[-1]}" + (f"_{MAX_EXAMPLES}" if MAX_EXAMPLES else "")\n\n'
         'adapter = train(MODEL_ID, docs_train, q_train, RUN, epochs=1, batch_size=1, grad_accum=16,\n'
         '                max_examples=MAX_EXAMPLES, save_steps=100)'),
    md("## Evaluate on dev (use dev for all tuning decisions)"),
    code('model, tok = load_model(MODEL_ID, adapter=f"{RUN}/final")\n'
         'eval_qs = q_dev                         # or sample(q_dev, 300) for a quick check\n'
         'preds = predict(model, tok, docs_dev, eval_qs, f"{RUN}/pred_dev.jsonl", batch_size=8)\n'
         'result = score(eval_qs, preds)\n'
         'print_report(result, "fine-tuned, dev")'),
    code(SHOW_ERRORS),
    md("## Final: test set. Run this only once you stop changing things\n"
       "It also writes `pred_test_official.json` in the official `{uid: [answer, scale]}` format."),
    code('import json\n'
         'preds_test = predict(model, tok, docs_test, q_test, f"{RUN}/pred_test.jsonl", batch_size=8)\n'
         'result_test = score(q_test, preds_test)\n'
         'print_report(result_test, "fine-tuned, TEST")\n'
         'with open(f"{RUN}/pred_test_official.json", "w") as f:\n'
         '    json.dump({u: [p["answer"], p["scale"]] for u, p in preds_test.items()}, f)'),
])

os.makedirs(os.path.join(ROOT, "notebooks"), exist_ok=True)
for name, nb in (("01_zero_shot_baseline.ipynb", baseline), ("02_finetune_and_eval.ipynb", finetune)):
    with open(os.path.join(ROOT, "notebooks", name), "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("wrote notebooks/" + name)
