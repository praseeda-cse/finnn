"""Generate the Colab notebooks in notebooks/. Edit the cell text here, then re-run.

    python scripts/make_notebooks.py
"""
import glob
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
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes", "accelerate"], check=True)
else:
    ROOT = os.path.abspath("..")
    OUT = os.path.join(ROOT, "runs")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, ROOT)

import json, torch
from finsight.data import load_split
from finsight.evaluate import score, print_report
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE - select a GPU runtime!")

docs_train, q_train = load_split(f"{ROOT}/data/tatdqa_train.json.gz")
docs_dev, q_dev = load_split(f"{ROOT}/data/tatdqa_dev.json.gz")
docs_test, q_test = load_split(f"{ROOT}/data/tatdqa_test.json.gz")
def sample(questions, n, seed=0):
    return random.Random(seed).sample(questions, n) if n < len(questions) else questions
def report_row(r):
    return {m: v for m, v in r.items() if m != "rows"}
print(len(q_train), len(q_dev), len(q_test), "questions")'''

SHOW_ERRORS = r'''# Look at mistakes: this is where the next improvement comes from
rows = {r["uid"]: r for r in result["rows"]}
wrong = [q for q in eval_qs if rows[q["uid"]]["em"] == 0]
print(len(wrong), "wrong of", len(eval_qs))
for q in wrong[:10]:
    p = preds[q["uid"]]
    print(f"[{q['answer_type']}] {q['question']}")
    print(f"   gold: {q['answer']} ({q['scale']}) | evidence rows {q['evidence']} | derivation {q['derivation']!r}")
    print(f"   pred: {p['answer']} ({p['scale']}) | evidence rows {p['evidence']}")
    print("   raw:", p["raw"].replace(chr(10), " / ")[:400], "\n")'''


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text}


def notebook(cells):
    return {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"accelerator": "GPU", "kernelspec": {"name": "python3", "display_name": "Python 3"},
                         "language_info": {"name": "python"}}}


baseline = notebook([
    md("# 01 · Baseline: a general-purpose LLM, no training\n\n"
       "Qwen2.5-7B-Instruct reads the page (numbered rows with ` | ` column breaks) and writes "
       "`evidence / type / formula / unit`; Python executes the formula; the **official TAT-DQA scorer** grades it.\n\n"
       "This is the comparison row for a general model that was never trained on financial documents. "
       "Runtime on a T4: about 15–25 min for 200 dev questions."),
    code(SETUP),
    code('from finsight.infer import load_model, predict\n\n'
         'MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"   # 4-bit, fits a 16 GB T4\n'
         'N = 200                                 # dev questions (random sample, seed 0, shared by all notebooks)\n'
         'eval_qs = sample(q_dev, N)\n'
         'name = MODEL_ID.split("/")[-1]\n\n'
         'model, tok = load_model(MODEL_ID)\n'
         'preds = predict(model, tok, docs_dev, eval_qs, f"{OUT}/zeroshot_{name}_dev{N}.jsonl", few_shot=True, batch_size=4)\n'
         'result = score(eval_qs, preds)\n'
         'print_report(result, f"{name} few-shot, dev sample")'),
    code(SHOW_ERRORS),
])

pipeline = notebook([
    md("# 02 · FinSight pipeline: TAT-LLM + executor + self-consistency (main results)\n\n"
       "**No training on our side.** Stage 2 uses TAT-LLM (Zhu et al., 2024), LLaMA-2 already fine-tuned by its "
       "authors on FinQA, TAT-QA and TAT-DQA (`next-tat/tat-llm-7b-fft`, public). Its prompt is rebuilt exactly as in "
       "the authors' released files.\n\n"
       "**Our additions, each measured separately with the official, unit-aware scorer:**\n"
       "1. **Python executor**: the model's equation is executed exactly instead of trusting its own arithmetic, with "
       "x100 reconciliation for percentages.\n"
       "2. **Self-consistency voting**: K sampled answers + the greedy answer are executed; the most common result wins.\n"
       "3. **Evidence grounding**: the model's evidence values are located in the page rows (evidence recall / precision).\n\n"
       "Part A uses the same 200 dev questions as notebook 01. Part C is the single final run on the official test set.\n\n"
       "Rough T4 time: Part A ≈ 30 min (model download ≈ 10 min the first time), Part B ≈ 1–1.5 h, Part C ≈ 3–4 h. "
       "Every part saves to Drive and resumes after a disconnect (re-run setup, the model cell, then the same cell)."),
    code(SETUP),
    code('from finsight import tatllm\n\n'
         'MODEL_ID = "next-tat/tat-llm-7b-fft"    # or "next-tat/tat-llm-13b-fft" (slower; use batch_size=2)\n'
         'name = MODEL_ID.split("/")[-1]\n'
         'model, tok = tatllm.load(MODEL_ID)'),
    md("## Part A · Greedy answers: model's own arithmetic vs. our executor (dev sample)"),
    code('N = 200\n'
         'eval_qs = sample(q_dev, N)                # same sample as notebook 01\n'
         'doc_of = {q["uid"]: docs_dev[q["doc_uid"]] for q in eval_qs}\n'
         'raw = tatllm.predict(model, tok, docs_dev, eval_qs, f"{OUT}/{name}_dev{N}.jsonl", batch_size=4)\n\n'
         'RESULTS = {}\n'
         'base_file = f"{OUT}/zeroshot_Qwen2.5-7B-Instruct_dev{N}.jsonl"\n'
         'if os.path.exists(base_file):\n'
         '    base = {json.loads(l)["uid"]: json.loads(l) for l in open(base_file)}\n'
         '    RESULTS["General LLM: Qwen2.5-7B, no training (notebook 01)"] = score(eval_qs, base)\n'
         'own = tatllm.score_outputs(raw, use_executor=False, doc_of=doc_of)\n'
         'exe = tatllm.score_outputs(raw, use_executor=True, doc_of=doc_of)\n'
         'RESULTS[f"{name}: model\'s own arithmetic (authors\' setup)"] = score(eval_qs, own)\n'
         'RESULTS[f"{name} + executor (ours)"] = score(eval_qs, exe)\n'
         'for label, r in RESULTS.items():\n'
         '    print_report(r, label)'),
    md("## Part B · Self-consistency voting (dev sample)"),
    code('K = 5\n'
         'samples = tatllm.sample(model, tok, docs_dev, eval_qs, f"{OUT}/{name}_dev{N}_k{K}.jsonl", k=K, batch_size=2)\n'
         'voted = {q["uid"]: tatllm.vote(samples[q["uid"]]["samples"], raw[q["uid"]]["raw"], doc_of[q["uid"]])\n'
         '         for q in eval_qs}\n'
         'RESULTS[f"{name} + executor + voting (k={K}) (ours)"] = score(eval_qs, voted)\n'
         'print_report(RESULTS[f"{name} + executor + voting (k={K}) (ours)"], "with voting")\n\n'
         'print("\\nSummary (official scorer, same 200 dev questions):")\n'
         'for label, r in RESULTS.items():\n'
         '    print(f"  {r[\'EM\']:6.2f} EM  {r[\'F1\']:6.2f} F1  unit {r[\'scale_acc\']:5.1f}  {label}")\n'
         'json.dump({k: report_row(r) for k, r in RESULTS.items()}, open(f"{OUT}/finsight_dev{N}_results.json", "w"), indent=1)'),
    code('# Mistakes of the best setup\n'
         'result, preds = RESULTS[f"{name} + executor + voting (k={K}) (ours)"], voted\n' + SHOW_ERRORS),
    md("## Part C · Final run on the official TEST set (run once, at the end)\n"
       "Greedy + executor on all 1,663 test questions (≈ 3–4 h on a T4). Set `VOTE = True` to add voting "
       "(≈ 5x longer; only if you have the GPU time). Writes `test_official.json` in the official `{uid: [answer, scale]}` format."),
    code('VOTE = False\n'
         'doc_test = {q["uid"]: docs_test[q["doc_uid"]] for q in q_test}\n'
         'raw_t = tatllm.predict(model, tok, docs_test, q_test, f"{OUT}/{name}_test.jsonl", batch_size=4)\n'
         'own_t = tatllm.score_outputs(raw_t, use_executor=False, doc_of=doc_test)\n'
         'final_t = tatllm.score_outputs(raw_t, use_executor=True, doc_of=doc_test)\n'
         'TEST = {f"{name}: model\'s own arithmetic": score(q_test, own_t), f"{name} + executor (ours)": score(q_test, final_t)}\n'
         'if VOTE:\n'
         '    s_t = tatllm.sample(model, tok, docs_test, q_test, f"{OUT}/{name}_test_k5.jsonl", k=5, batch_size=2)\n'
         '    final_t = {q["uid"]: tatllm.vote(s_t[q["uid"]]["samples"], raw_t[q["uid"]]["raw"], doc_test[q["uid"]]) for q in q_test}\n'
         '    TEST[f"{name} + executor + voting (ours)"] = score(q_test, final_t)\n'
         'for label, r in TEST.items():\n'
         '    print_report(r, "TEST: " + label)\n'
         'json.dump({u: [p["answer"], p["scale"]] for u, p in final_t.items()}, open(f"{OUT}/test_official.json", "w"))\n'
         'json.dump({k: report_row(r) for k, r in TEST.items()}, open(f"{OUT}/finsight_test_results.json", "w"), indent=1)\n\n'
         'print("\\nPublished TAT-DQA test results for comparison (EM): MHST 41.5 · Doc2SoarGraph 59.2 · GPT-4 64.5 · "\n'
         '      "TAT-LLM-7B 72.6 (authors\' lenient evaluation script)")'),
])

demo = notebook([
    md("# 03 · FinSight demo app\n\n"
       "Upload a digital financial PDF (up to 3 pages) and ask a question. The app shows the answer and unit, the "
       "equation Python executed, and the page with the evidence rows highlighted.\n\n"
       "Test PDFs: any file from `fp/dataset/tatdqa_docs_test/test/*.pdf` on your laptop."),
    code(SETUP),
    code('from finsight import tatllm\n'
         'model, tok = tatllm.load("next-tat/tat-llm-7b-fft")'),
    code('subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gradio"], check=True)\n'
         'import gradio as gr\n'
         'from finsight.demo import doc_from_pdf, answer, highlight_pages, explain_html\n\n'
         'def run(pdf, question):\n'
         '    if pdf is None or not question.strip():\n'
         '        return "<p>Upload a PDF and type a question.</p>", []\n'
         '    path = pdf if isinstance(pdf, str) else pdf.name\n'
         '    doc, scales = doc_from_pdf(path)\n'
         '    res = answer(model, tok, doc, question)\n'
         '    return explain_html(res, doc), highlight_pages(path, doc, scales, res["evidence"])\n\n'
         'app = gr.Interface(\n'
         '    fn=run,\n'
         '    inputs=[gr.File(label="Financial PDF (digital, up to 3 pages)", file_types=[".pdf"]),\n'
         '            gr.Textbox(label="Question", placeholder="What was the percentage change in revenue from 2018 to 2019?")],\n'
         '    outputs=[gr.HTML(label="Answer"), gr.Gallery(label="Evidence on the page", columns=1, height=800)],\n'
         '    title="FinSight: grounded question answering on financial documents",\n'
         '    description="TAT-LLM reads the page and writes an equation; Python computes the answer; evidence is highlighted.")\n'
         'app.launch(share=True)   # opens inside the notebook; the public link works for about 72 h'),
])

nb_dir = os.path.join(ROOT, "notebooks")
os.makedirs(nb_dir, exist_ok=True)
for old in glob.glob(os.path.join(nb_dir, "*.ipynb")):
    os.remove(old)
for name, nb in (("01_baseline_general_llm.ipynb", baseline), ("02_finsight_pipeline.ipynb", pipeline),
                 ("03_demo_app.ipynb", demo)):
    with open(os.path.join(nb_dir, name), "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("wrote notebooks/" + name)
