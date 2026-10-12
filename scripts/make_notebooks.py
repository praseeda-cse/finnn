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
def latest_adapter(run):
    """The finished adapter, or the newest checkpoint if training stopped midway."""
    import glob
    if os.path.exists(f"{run}/final"):
        return f"{run}/final"
    ckpts = sorted(glob.glob(f"{run}/checkpoint-*"), key=lambda p: int(p.rsplit("-", 1)[1]))
    assert ckpts, f"No adapter in {run} - run notebook 02 (training) first"
    print("NOTE: training has not finished; using", ckpts[-1])
    return ckpts[-1]
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
    md("## Evaluate on dev (use dev for all tuning decisions)\n"
       "Uses the finished adapter, **or the newest checkpoint** if training stopped. To evaluate a checkpoint "
       "without training further, run only the setup cell and then this cell (skip the training cell)."),
    code('MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"\n'
         'RUN = f"{OUT}/sft_Qwen2.5-3B-Instruct"\n'
         'ADAPTER = latest_adapter(RUN)\n'
         'N_EVAL = 300                            # quick check (~10 min); len(q_dev) = 1,644 for the full dev score\n'
         'eval_qs = sample(q_dev, N_EVAL, seed=1)\n'
         'tag = os.path.basename(ADAPTER)\n'
         'model, tok = load_model(MODEL_ID, adapter=ADAPTER)\n'
         'preds = predict(model, tok, docs_dev, eval_qs, f"{RUN}/pred_dev_{tag}_{N_EVAL}.jsonl", batch_size=8)\n'
         'result = score(eval_qs, preds)\n'
         'print_report(result, f"fine-tuned ({tag}), dev sample")'),
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

LOAD_SFT = ('MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"\n'
            'SFT_RUN = f"{OUT}/sft_Qwen2.5-3B-Instruct"\n'
            'ADAPTER = latest_adapter(SFT_RUN)\n'
            'model, tok = load_model(MODEL_ID, adapter=ADAPTER)')

trust = notebook([
    md("# 03 · TRUST and REINFORCE: voting, confidence, abstention and expert iteration\n\n"
       "Needs the fine-tuned adapter from notebook 02 (the newest checkpoint is used if training stopped midway).\n\n"
       "**Part A (Steps 5.4–5.7):** sample K answers per question, vote with weights from the verifier, turn the vote "
       "into a confidence score, and choose a threshold that keeps the error rate on *answered* questions below a target.\n\n"
       "**Part B (Steps 4.2, 4.5):** expert iteration. Sample answers on training questions, keep only those that are "
       "correct **and** cite the right evidence (the evidence-aware reward), and fine-tune on them.\n\n"
       "Rough T4 timing: Part A ≈ 40–60 min, Part B ≈ 1.5–2 h. Every step saves to Drive and resumes after a disconnect."),
    code(SETUP),
    code(LOAD_SFT),
    md("## Part A · Greedy vs. verification-weighted voting (dev sample)"),
    code('from finsight.infer import sample as draw_samples\n'
         'from finsight.trust import vote, risk_coverage, choose_threshold, selective_report\n'
         'from finsight.reinforce import answer_correct, evidence_f1\n\n'
         'N, K = 500, 5\n'
         'eval_qs = sample(q_dev, N, seed=1)\n'
         'greedy = predict(model, tok, docs_dev, eval_qs, f"{SFT_RUN}/greedy_dev{N}.jsonl", batch_size=8)\n'
         'print_report(score(eval_qs, greedy), "fine-tuned, greedy")\n\n'
         'raw = draw_samples(model, tok, docs_dev, eval_qs, f"{SFT_RUN}/samples_dev{N}_k{K}.jsonl", k=K, batch_size=2)\n'
         'voted = {q["uid"]: vote(raw[q["uid"]], q, docs_dev[q["doc_uid"]]) for q in eval_qs}\n'
         'print_report(score(eval_qs, voted), f"fine-tuned, verification-weighted vote (k={K})")'),
    md("## Confidence and selective answering\n"
       "The threshold is chosen on one half of the sample and checked on the other half, so the reported error rate is honest."),
    code('import matplotlib.pyplot as plt\n'
         'conf = [voted[q["uid"]]["confidence"] for q in eval_qs]\n'
         'ok = [int(answer_correct(voted[q["uid"]]["answer"], voted[q["uid"]]["scale"], q)) for q in eval_qs]\n'
         'half = len(eval_qs) // 2\n'
         'for target in (0.05, 0.10, 0.20):\n'
         '    t = choose_threshold(conf[:half], ok[:half], target)\n'
         '    print(f"target error <= {target:.0%}:", selective_report(conf[half:], ok[half:], t) if t is not None else "not reachable on this sample")\n\n'
         'rc = risk_coverage(conf, ok)\n'
         'plt.figure(figsize=(6, 4))\n'
         'plt.plot([c * 100 for c, _, _ in rc], [a * 100 for _, a, _ in rc], marker=".")\n'
         'plt.xlabel("coverage: % of questions answered"); plt.ylabel("accuracy on answered questions (%)")\n'
         'plt.title("Risk-coverage curve (dev sample)"); plt.grid(alpha=.3)\n'
         'plt.savefig(f"{SFT_RUN}/risk_coverage.png", dpi=150, bbox_inches="tight"); plt.show()'),
    md("## Part B · Expert iteration (light-weight reinforcement learning)\n"
       "Keeps only samples that are correct, cite evidence with F1 ≥ 0.5 and whose numbers appear in the cited "
       "evidence. Correct answers built on the wrong evidence (*right for the wrong reason*) are counted and thrown away."),
    code('import json\n'
         'from finsight.reinforce import select_targets\n\n'
         'N_TRAIN, K_TRAIN = 1000, 4\n'
         'ei_qs = sample(q_train, N_TRAIN, seed=2)\n'
         'raw_tr = draw_samples(model, tok, docs_train, ei_qs, f"{SFT_RUN}/samples_train{N_TRAIN}_k{K_TRAIN}.jsonl",\n'
         '                      k=K_TRAIN, batch_size=2)\n'
         'targets, stats = select_targets(ei_qs, docs_train, raw_tr)\n'
         'print(stats)\n'
         'json.dump(targets, open(f"{SFT_RUN}/ei_targets.json", "w"))'),
    code('import gc\n'
         'from finsight.train import train\n'
         'del model; gc.collect(); torch.cuda.empty_cache()\n\n'
         'EI_RUN = f"{OUT}/ei1_Qwen2.5-3B-Instruct"\n'
         'ei_adapter = train(MODEL_ID, docs_train, ei_qs, EI_RUN, epochs=1, lr=5e-5, batch_size=1, grad_accum=8,\n'
         '                   targets=targets, init_adapter=ADAPTER, save_steps=50)'),
    code('model, tok = load_model(MODEL_ID, adapter=ei_adapter)\n'
         'greedy_ei = predict(model, tok, docs_dev, eval_qs, f"{EI_RUN}/greedy_dev{N}.jsonl", batch_size=8)\n'
         'print_report(score(eval_qs, greedy), "SFT, greedy")\n'
         'print_report(score(eval_qs, greedy_ei), "SFT + expert iteration, greedy")\n\n'
         'def wrong_reason_rate(preds):\n'
         '    right = [q for q in eval_qs if answer_correct(preds[q["uid"]]["answer"], preds[q["uid"]]["scale"], q)]\n'
         '    bad = [q for q in right if evidence_f1(preds[q["uid"]]["evidence"], q["evidence"]) < 0.5]\n'
         '    return 100 * len(bad) / max(1, len(right))\n'
         'print(f"right-for-the-wrong-reason: SFT {wrong_reason_rate(greedy):.1f}% -> SFT+EI {wrong_reason_rate(greedy_ei):.1f}% of correct answers")'),
])

demo = notebook([
    md("# 04 · FinSight demo app\n\n"
       "Upload a digital financial PDF (up to 3 pages) and ask a question. The app shows the answer, the formula "
       "Python executed, the verifier checks, and the page with the evidence rows highlighted.\n\n"
       "Test PDFs: any file from `fp/dataset/tatdqa_docs_test/test/*.pdf` on your laptop."),
    code(SETUP),
    code(LOAD_SFT + '\n# To demo the expert-iteration model instead:\n# model, tok = load_model(MODEL_ID, adapter=latest_adapter(f"{OUT}/ei1_Qwen2.5-3B-Instruct"))'),
    code('subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gradio"], check=True)\n'
         'import gradio as gr\n'
         'from finsight.demo import doc_from_pdf, answer, highlight_pages, explain_html\n\n'
         'def run(pdf, question):\n'
         '    if pdf is None or not question.strip():\n'
         '        return "<p>Upload a PDF and type a question.</p>", []\n'
         '    path = pdf if isinstance(pdf, str) else pdf.name\n'
         '    doc, scales = doc_from_pdf(path)\n'
         '    res = answer(model, tok, doc, question)\n'
         '    return explain_html(res, doc), highlight_pages(path, doc, scales, res["parsed"]["evidence"])\n\n'
         'app = gr.Interface(\n'
         '    fn=run,\n'
         '    inputs=[gr.File(label="Financial PDF (digital, up to 3 pages)", file_types=[".pdf"]),\n'
         '            gr.Textbox(label="Question", placeholder="What was the percentage change in revenue from 2018 to 2019?")],\n'
         '    outputs=[gr.HTML(label="Answer"), gr.Gallery(label="Evidence on the page", columns=1, height=800)],\n'
         '    title="FinSight: grounded question answering on financial documents",\n'
         '    description="The model cites evidence rows and writes a formula; Python computes the answer.")\n'
         'app.launch(share=True)   # opens inside the notebook; the public link works for about 72 h'),
])

nofinetune = notebook([
    md("# 05 · No fine-tuning: model comparison with retrieved examples\n\n"
       "No training at all. For every question, the 4 most similar **training** questions are shown to the model as "
       "solved examples (their evidence rows, unit rows and correct 4-line answer). The model learns the dataset's "
       "answer-type and unit conventions from them. Python still computes every number.\n\n"
       "Same 200 dev questions as notebook 01 (where Qwen2.5-7B with 2 fixed examples scored **42.0 EM**), so the "
       "numbers are directly comparable.\n\n"
       "Run the cells **one at a time**. Each model cell takes roughly 20–60 min on a T4 and resumes after a "
       "disconnect (re-run setup, the helper cell, then the same model cell)."),
    code(SETUP),
    code('import gc, json\n'
         'from finsight.examples import ExampleRetriever\n\n'
         'N = 200\n'
         'eval_qs = sample(q_dev, N)                 # same sample as notebook 01 (seed 0)\n'
         'retriever = ExampleRetriever(docs_train, q_train)\n'
         'examples_fn = lambda q: retriever(q, k=4)\n'
         'RESULTS = {}\n'
         'base_file = f"{OUT}/zeroshot_Qwen2.5-7B-Instruct_dev{N}.jsonl"\n'
         'if os.path.exists(base_file):\n'
         '    base = {json.loads(l)["uid"]: json.loads(l) for l in open(base_file)}\n'
         '    RESULTS["Qwen2.5-7B, 2 fixed examples (notebook 01)"] = score(eval_qs, base)\n\n'
         'def run(model_id, batch_size=4, template_kwargs=None, label=None):\n'
         '    name = model_id.split("/")[-1]\n'
         '    model, tok = load_model(model_id)\n'
         '    preds = predict(model, tok, docs_dev, eval_qs, f"{OUT}/noft_{name}_retrieved4_dev{N}.jsonl",\n'
         '                    batch_size=batch_size, examples_fn=examples_fn, template_kwargs=template_kwargs)\n'
         '    res = score(eval_qs, preds)\n'
         '    RESULTS[label or f"{name}, 4 retrieved examples"] = res\n'
         '    print_report(res, label or name)\n'
         '    del model; gc.collect(); torch.cuda.empty_cache()\n'
         '    return preds'),
    md("## Model A · Qwen2.5-7B-Instruct + retrieved examples\n"
       "Same model as notebook 01, so this shows what the retrieved examples alone add."),
    code('preds_a = run("Qwen/Qwen2.5-7B-Instruct", batch_size=4)'),
    md("## Model B · Qwen3-8B (thinking turned off)"),
    code('preds_b = run("Qwen/Qwen3-8B", batch_size=4, template_kwargs={"enable_thinking": False})'),
    md("## Model C · Qwen2.5-14B-Instruct (largest that fits a free T4 in 4-bit; slowest)"),
    code('preds_c = run("Qwen/Qwen2.5-14B-Instruct", batch_size=2)'),
    md("## Comparison table"),
    code('rows = []\n'
         'for label, r in RESULTS.items():\n'
         '    rows.append([label, f"{r[\'EM\']:.1f}", f"{r[\'F1\']:.1f}", f"{r[\'scale_acc\']:.1f}",\n'
         '                 f"{r.get(\'arithmetic_EM\', 0):.1f}", f"{r.get(\'span_EM\', 0):.1f}",\n'
         '                 f"{r[\'evidence_recall\']:.0f}/{r[\'evidence_precision\']:.0f}"])\n'
         'head = ["setup", "EM", "F1", "unit acc", "arith EM", "span EM", "evid R/P"]\n'
         'w = [max(len(str(x)) for x in col) for col in zip(head, *rows)]\n'
         'for row in [head] + rows:\n'
         '    print("  ".join(str(x).ljust(n) for x, n in zip(row, w)))\n'
         'json.dump({k: {m: v for m, v in r.items() if m != "rows"} for k, r in RESULTS.items()},\n'
         '          open(f"{OUT}/noft_comparison_dev{N}.json", "w"), indent=1)'),
])

tatllm_nb = notebook([
    md("# 06 · TAT-LLM (pre-trained by its authors) + our executor, scored with the official scorer\n\n"
       "**No training on our side.** TAT-LLM (Zhu et al., 2024) is LLaMA-2 already fine-tuned by its authors on "
       "FinQA, TAT-QA and TAT-DQA (`next-tat/tat-llm-7b-fft`, public). Its prompt is rebuilt exactly as in the "
       "authors' released files (identical for all 1,644 dev questions).\n\n"
       "The authors report 72.6 EM on TAT-DQA, but their evaluation script accepts answers within ~1% (or off by "
       "x100) and ignores the predicted unit. Here every answer is graded by the **official, unit-aware scorer**, "
       "twice: with the model's own arithmetic, and with **our Python executor** running the model's equation.\n\n"
       "Same 200 dev questions as notebooks 01 and 05. Rough time on a T4: model download ~5 min, answering ~25 min."),
    code(SETUP),
    code('import json\n'
         'from finsight import tatllm\n\n'
         'MODEL_ID = "next-tat/tat-llm-7b-fft"    # or "next-tat/tat-llm-13b-fft" (slower, use batch_size=2)\n'
         'N = 200\n'
         'eval_qs = sample(q_dev, N)                # same sample as notebooks 01 and 05\n'
         'name = MODEL_ID.split("/")[-1]\n'
         'model, tok = tatllm.load(MODEL_ID)\n'
         'raw = tatllm.predict(model, tok, docs_dev, eval_qs, f"{OUT}/{name}_dev{N}.jsonl", batch_size=4)'),
    md("## Official scores: model's own arithmetic vs. our executor"),
    code('RESULTS = {}\n'
         'base_file = f"{OUT}/zeroshot_Qwen2.5-7B-Instruct_dev{N}.jsonl"\n'
         'if os.path.exists(base_file):\n'
         '    base = {json.loads(l)["uid"]: json.loads(l) for l in open(base_file)}\n'
         '    RESULTS["Qwen2.5-7B untrained, our format (notebook 01)"] = score(eval_qs, base)\n'
         'own = tatllm.score_outputs(raw, use_executor=False)\n'
         'exe = tatllm.score_outputs(raw, use_executor=True)\n'
         'RESULTS[f"{name}, model\'s own arithmetic"] = score(eval_qs, own)\n'
         'RESULTS[f"{name} + our Python executor"] = score(eval_qs, exe)\n'
         'for label, r in RESULTS.items():\n'
         '    print_report(r, label)\n'
         'json.dump({k: {m: v for m, v in r.items() if m != "rows"} for k, r in RESULTS.items()},\n'
         '          open(f"{OUT}/{name}_comparison_dev{N}.json", "w"), indent=1)'),
    md("## Look at mistakes (with the executor)"),
    code('rows = {r["uid"]: r for r in RESULTS[f"{name} + our Python executor"]["rows"]}\n'
         'wrong = [q for q in eval_qs if rows[q["uid"]]["em"] == 0]\n'
         'print(len(wrong), "wrong of", len(eval_qs))\n'
         'for q in wrong[:10]:\n'
         '    p = exe[q["uid"]]\n'
         '    print(f"[{q[\'answer_type\']}] {q[\'question\']}")\n'
         '    print(f"   gold: {q[\'answer\']} ({q[\'scale\']}) | derivation {q[\'derivation\']!r}")\n'
         '    print(f"   pred: {p[\'answer\']} ({p[\'scale\']})")\n'
         '    print("   raw:", p["raw"].replace(chr(10), " / ")[:400], "\\n")'),
])

os.makedirs(os.path.join(ROOT, "notebooks"), exist_ok=True)
for name, nb in (("01_zero_shot_baseline.ipynb", baseline), ("02_finetune_and_eval.ipynb", finetune),
                 ("03_trust_and_reinforce.ipynb", trust), ("04_demo_app.ipynb", demo),
                 ("05_no_finetune_model_comparison.ipynb", nofinetune),
                 ("06_tatllm_with_executor.ipynb", tatllm_nb)):
    with open(os.path.join(ROOT, "notebooks", name), "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("wrote notebooks/" + name)
