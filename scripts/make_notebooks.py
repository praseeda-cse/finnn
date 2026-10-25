"""Generate the Colab notebooks in notebooks/. Edit the cell text here, then re-run.

    python scripts/make_notebooks.py
"""
import glob
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

SETUP = r'''# --- Setup: on Colab, gets the latest code + data from GitHub; also works on a local GPU ---
import os, sys, subprocess, shutil, random
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # less GPU memory fragmentation
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
         'MODEL_ID = "next-tat/tat-llm-7b-fft"    # or "next-tat/tat-llm-13b-fft" (slower; then set tatllm.TOKEN_BUDGET = 5000)\n'
         'name = MODEL_ID.split("/")[-1]\n'
         'model, tok = tatllm.load(MODEL_ID)'),
    md("## Part A · Greedy answers: model's own arithmetic vs. our executor (dev sample)"),
    code('N = 200\n'
         'eval_qs = sample(q_dev, N)                # same sample as notebook 01\n'
         'doc_of = {q["uid"]: docs_dev[q["doc_uid"]] for q in eval_qs}\n'
         'raw = tatllm.predict(model, tok, docs_dev, eval_qs, f"{OUT}/{name}_dev{N}.jsonl")\n\n'
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
         'samples = tatllm.sample(model, tok, docs_dev, eval_qs, f"{OUT}/{name}_dev{N}_k{K}.jsonl", k=K)\n'
         'voted = {q["uid"]: tatllm.vote(samples[q["uid"]]["samples"], raw[q["uid"]]["raw"], doc_of[q["uid"]])\n'
         '         for q in eval_qs}\n'
         'RESULTS[f"{name} + executor + voting (k={K}) (ours)"] = score(eval_qs, voted)\n'
         'print_report(RESULTS[f"{name} + executor + voting (k={K}) (ours)"], "with voting")\n\n'
         'print("\\nSummary (official scorer, same 200 dev questions):")\n'
         'for label, r in RESULTS.items():\n'
         '    print(f"  {r[\'EM\']:6.2f} EM  {r[\'F1\']:6.2f} F1  unit {r[\'scale_acc\']:5.1f}  {label}")\n'
         'json.dump({k: report_row(r) for k, r in RESULTS.items()}, open(f"{OUT}/finsight_dev{N}_results.json", "w"), indent=1)'),
    md("## Error breakdown: which fix would gain the most?\n"
       "For every wrong answer, checks whether fixing only the unit, the sign, a x100 factor or rounding would make "
       "it correct. Send this table: it decides the next improvement."),
    code('from finsight.analysis import error_breakdown, print_breakdown\n'
         'for label, p in ((f"{name} + executor", exe), ("+ voting", voted)):\n'
         '    r = score(eval_qs, p)\n'
         '    print(f"\\n=== {label} ===")\n'
         '    print_breakdown(error_breakdown(eval_qs, p, r["rows"]), len(eval_qs))'),
    code('# Mistakes of the best setup\n'
         'result, preds = RESULTS[f"{name} + executor + voting (k={K}) (ours)"], voted\n' + SHOW_ERRORS),
    md("## Part C · Final run on the official TEST set (run once, at the end)\n"
       "**Switched off by default** so that *Run all* is safe. When the dev results are final, set `RUN_TEST = True` "
       "and run this cell. Greedy + executor on all 1,663 test questions (≈ 3–4 h on a T4). Set `VOTE = True` to add "
       "voting (≈ 5x longer). Writes `test_official.json` in the official `{uid: [answer, scale]}` format."),
    code('RUN_TEST = False\n'
         'VOTE = False\n'
         'if not RUN_TEST:\n'
         '    raise SystemExit("Part C skipped (RUN_TEST = False). The dev results above are complete.")\n'
         'doc_test = {q["uid"]: docs_test[q["doc_uid"]] for q in q_test}\n'
         'raw_t = tatllm.predict(model, tok, docs_test, q_test, f"{OUT}/{name}_test.jsonl")\n'
         'own_t = tatllm.score_outputs(raw_t, use_executor=False, doc_of=doc_test)\n'
         'final_t = tatllm.score_outputs(raw_t, use_executor=True, doc_of=doc_test)\n'
         'TEST = {f"{name}: model\'s own arithmetic": score(q_test, own_t), f"{name} + executor (ours)": score(q_test, final_t)}\n'
         'if VOTE:\n'
         '    s_t = tatllm.sample(model, tok, docs_test, q_test, f"{OUT}/{name}_test_k5.jsonl", k=5)\n'
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

paper1 = notebook([
    md("# 04 · Paper 1 follow-up: does the VLM fail to read, or fail to answer?\n\n"
       "Qwen2-VL-2B on the **same 200 test questions** and the **same 150-DPI page images** as the original run, "
       "scored with the official TAT-DQA scorer. Conditions:\n"
       "- **A** original run (96 output tokens, prompt chosen by the gold answer type), loaded from file, no GPU\n"
       "- **B** same prompts, **512** output tokens: does a longer budget fix arithmetic?\n"
       "- **C** one **generic** prompt for every question (no gold information), 512 tokens\n"
       "- **D** **structured** output: `Answer:` (value, or an arithmetic expression executed by Python) + `Unit:`\n\n"
       "Runtime → Change runtime type → **T4 GPU** (L4/A100 is faster). Rough T4 time: 1.5–2.5 h in total; every "
       "condition saves to Drive and resumes after a disconnect (re-run setup + model cell, then the same cell). "
       "At the end, send `paper1_rerun_results.json` from `My Drive/finsight/runs/`."),
    code(SETUP),
    code('# Page images (150 DPI, as in the original run) + original outputs.\n'
         '# Upload tatdqa_test_pages_150dpi.zip (83 MB, from the laptop: Desktop/oneee/final_year/) to My Drive/finsight/\n'
         'import zipfile\n'
         'PAGES = "/content/tatdqa_pages" if IN_COLAB else os.path.join(ROOT, "runs", "tatdqa_pages")\n'
         'ZIP = "/content/drive/MyDrive/finsight/tatdqa_test_pages_150dpi.zip"\n'
         'if not os.path.exists(f"{PAGES}/page_of_doc.json"):\n'
         '    assert os.path.exists(ZIP), "Upload tatdqa_test_pages_150dpi.zip to My Drive/finsight/ first"\n'
         '    zipfile.ZipFile(ZIP).extractall(PAGES)\n'
         'page_of = json.load(open(f"{PAGES}/page_of_doc.json"))\n'
         'old_raw = json.load(open(f"{PAGES}/old_run_raw.json", encoding="utf-8"))\n\n'
         'N = 200\n'
         'eval_qs = sample(q_test, N)                       # fixed sample (seed 0)\n'
         'image_of = {q["uid"]: f"{PAGES}/pages/{q[\'doc_uid\']}_{page_of[q[\'doc_uid\']]}.png" for q in eval_qs}\n'
         'gold_scale = {q["uid"]: q["scale"] for q in eval_qs}\n'
         'print(len(eval_qs), "questions,", sum(os.path.exists(p) for p in image_of.values()), "page images found")'),
    code('from finsight import vlm\n'
         'model, processor = vlm.load()\n'
         '# quick check on 2 questions before the long runs\n'
         'print(vlm.run(model, processor, eval_qs[:2], image_of, "structured", "/tmp/check.jsonl", 64, batch_size=2))'),
    code('RAW = {"A": {q["uid"]: old_raw[q["uid"]] for q in eval_qs}}\n'
         'RAW["B"] = vlm.run(model, processor, eval_qs, image_of, "orig", f"{OUT}/p1_B_orig512.jsonl", 512)\n'
         'RAW["C"] = vlm.run(model, processor, eval_qs, image_of, "generic", f"{OUT}/p1_C_generic512.jsonl", 512)\n'
         'RAW["D"] = vlm.run(model, processor, eval_qs, image_of, "structured", f"{OUT}/p1_D_structured.jsonl", 128)'),
    code('MODE = {"A": "orig", "B": "orig", "C": "generic", "D": "structured"}\n'
         'NAME = {"A": "original (96 tokens, gold-type prompt)", "B": "512 tokens, gold-type prompt",\n'
         '        "C": "512 tokens, generic prompt", "D": "structured: expression executed + unit"}\n'
         'SUMMARY = {}\n'
         'for k in "ABCD":\n'
         '    p = vlm.to_preds(RAW[k], MODE[k])\n'
         '    r = score(eval_qs, p)\n'
         '    r_gold = score(eval_qs, vlm.to_preds(RAW[k], MODE[k], gold_scale))\n'
         '    print_report(r, f"{k}: {NAME[k]}")\n'
         '    SUMMARY[k] = {"name": NAME[k], "official": report_row(r), "gold_scale": report_row(r_gold),\n'
         '                  "arithmetic_behaviour": vlm.behaviour(eval_qs, RAW[k], p),\n'
         '                  "examples": [{"q": q["question"], "type": q["answer_type"], "gold": q["answer"], "scale": q["scale"],\n'
         '                                "raw": RAW[k][q["uid"]][:600]} for q in eval_qs[:15]]}\n'
         '    print("   arithmetic behaviour:", SUMMARY[k]["arithmetic_behaviour"], "\\n")\n'
         'SUMMARY["gpu"] = torch.cuda.get_device_name(0)\n'
         'json.dump(SUMMARY, open(f"{OUT}/paper1_rerun_results.json", "w"), indent=1, ensure_ascii=False)\n'
         'print("saved", f"{OUT}/paper1_rerun_results.json")\n\n'
         'print("\\nEM (official) / EM with gold unit:")\n'
         'for k in "ABCD":\n'
         '    s = SUMMARY[k]\n'
         '    print(f"  {k}  {s[\'official\'][\'EM\']:5.1f} / {s[\'gold_scale\'][\'EM\']:5.1f}   {s[\'name\']}")'),
])

SETUP_FINAL = r'''# --- Setup: Kaggle (recommended, 2 x T4) or Colab (1 x T4: models up to ~14B) ---
import os, sys, subprocess, shutil, random, json, time, glob
IN_KAGGLE = os.path.exists("/kaggle/working")
IN_COLAB = os.path.exists("/content") and not IN_KAGGLE
BASE = "/kaggle/working" if IN_KAGGLE else "/content" if IN_COLAB else os.path.abspath("..")
ROOT = f"{BASE}/finnn" if (IN_KAGGLE or IN_COLAB) else BASE
if IN_KAGGLE or IN_COLAB:
    shutil.rmtree(ROOT, ignore_errors=True)               # always fetch the latest code
    subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/praseeda-cse/finnn.git", ROOT], check=True)
OUT = f"{BASE}/runs"
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    OUT = "/content/drive/MyDrive/finsight/runs"
os.makedirs(OUT, exist_ok=True)
if IN_KAGGLE:                                             # earlier results uploaded as a Kaggle dataset
    for f in glob.glob("/kaggle/input/*/*.jsonl") + glob.glob("/kaggle/input/*/*/*.jsonl"):
        if not os.path.exists(f"{OUT}/{os.path.basename(f)}"):
            shutil.copy(f, OUT)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pymupdf", "gradio"], check=True)
sys.path.insert(0, ROOT)

from finsight.data import load_split
from finsight.evaluate import score, print_report
docs_train, q_train = load_split(f"{ROOT}/data/tatdqa_train.json.gz")
docs_dev, q_dev = load_split(f"{ROOT}/data/tatdqa_dev.json.gz")
docs_test, q_test = load_split(f"{ROOT}/data/tatdqa_test.json.gz")
def sample(questions, n, seed=0):
    return random.Random(seed).sample(questions, n) if n < len(questions) else questions
def read_jsonl(path):
    return {json.loads(l)["uid"]: json.loads(l) for l in open(path, encoding="utf-8")} if os.path.exists(path) else {}
def report_row(r):
    return {m: v for m, v in r.items() if m != "rows"}
print(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout)
print("results folder:", OUT, "|", sorted(os.listdir(OUT))[:20])'''

SERVER = r'''# --- Open models served locally by Ollama (llama.cpp inside; works on T4 GPUs; no API key, no cost) ---
import requests
from finsight.llm import ChatClient
PARALLEL = 2          # requests at once: 2 for ~32B models on 2 x T4, 4 for <=14B models
THINK = False         # Qwen3 models: False adds /no_think (much faster); True lets them reason first
if shutil.which("ollama") is None:
    subprocess.run("apt-get -qq update && apt-get -qq install -y zstd pciutils > /dev/null", shell=True)  # installer needs zstd
    subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True, check=True)
env = dict(os.environ, OLLAMA_CONTEXT_LENGTH="8192", OLLAMA_NUM_PARALLEL=str(PARALLEL), OLLAMA_KEEP_ALIVE="-1")
server = subprocess.Popen(["ollama", "serve"], env=env, stdout=open(f"{BASE}/ollama.log", "a"), stderr=subprocess.STDOUT)
for _ in range(60):
    try:
        requests.get("http://127.0.0.1:11434", timeout=2); break
    except Exception:
        time.sleep(1)

def use_model(model):
    """Download the model once (about 3-5 min for ~20 GB on Kaggle), unload any other model so the GPUs
    are free, and return a client for it."""
    loaded = subprocess.run(["ollama", "ps"], capture_output=True, text=True).stdout.splitlines()[1:]
    for name in (line.split()[0] for line in loaded if line.strip()):
        if name != model and name != model + ":latest":
            subprocess.run(["ollama", "stop", name])
    subprocess.run(["ollama", "pull", model], check=True, stdout=subprocess.DEVNULL)
    return ChatClient("http://127.0.0.1:11434/v1", model, workers=PARALLEL)

def prep(model, msgs):
    if "qwen3" in model and not THINK:
        msgs = [dict(m, content=m["content"] + " /no_think") if m["role"] == "system" else m for m in msgs]
    return msgs

def tag(model):
    return model.replace(":", "-").replace("/", "-")
print("Ollama is running")'''

final = notebook([
    md("# 05 · FinSight final pipeline: example RAG + best open model + executor + ensemble + risk report\n\n"
       "**No training anywhere.** Open models only, run for free on Kaggle.\n\n"
       "1. **Example RAG**: for every question, the 4 most similar *solved training questions* (their rows, gold "
       "program and unit) are put in the prompt. Retrieval was tuned on dev with `scripts/retrieval_check.py`.\n"
       "2. **Open LLM** writes evidence rows, answer type, a program (or the copied span) and the unit.\n"
       "3. **Python executes** the program; numbers and spans are checked against the page (grounding).\n"
       "4. **Ensemble**: where the main model and TAT-LLM-7B (notebook 02) disagree, the main model is sampled 4 "
       "more times and a weighted vote decides. The vote share is the confidence.\n"
       "5. **Automatic risk report**: line items are read by the model, checked against their rows, and ratios are "
       "computed in Python.\n\n"
       "**Target:** best published TAT-DQA test result is TAT-LLM-70B, **76.5 EM / 83.9 F1** (official leaderboard).\n\n"
       "**Kaggle settings (right panel):** Accelerator **GPU T4 x2** · Internet **on** (needs a phone-verified "
       "account) · Persistence **Files only** (so `runs/` survives a restart). Before starting, upload the notebook-02 "
       "outputs `tat-llm-7b-fft_dev200.jsonl` and `tat-llm-7b-fft_test.jsonl` from `My Drive/finsight/runs/` with "
       "*Add data → Upload* (any dataset name); they are copied into `runs/` automatically.\n\n"
       "Every run writes to `runs/` as it goes and resumes after a disconnect: re-run the setup and server cells, then "
       "the same cell. Rough times on 2 x T4 for a ~32B model: 30–40 min per 200 dev questions, 4–6 h for the test set."),
    code(SETUP_FINAL),
    code(SERVER),
    md("## Part A · Pick the most accurate open model (200 dev questions)\n"
       "Every candidate gets the same prompt, the same 4 retrieved examples and the same scorer. Start with 2–3 "
       "models; add newer open models from [ollama.com/library](https://ollama.com/library) if they fit "
       "(≤ ~22 GB download for 2 x T4, ≤ ~10 GB for one T4)."),
    code(r'''from finsight import solve
from finsight.retrieve import ExampleRetriever
retriever = ExampleRetriever(docs_train, q_train)   # example RAG over the TRAINING split only
N = 200
eval_qs = sample(q_dev, N)                           # same 200 dev questions as notebooks 01 and 02
RAW, RESULTS = {}, {}

def run(model, qs, docs, split, k=4, mode="direct", n=1, temperature=0.0):
    """Greedy (n=1) or sampled answers for qs; saved to runs/ and resumable."""
    client = use_model(model)
    jobs = {q["uid"]: prep(model, solve.messages(docs[q["doc_uid"]], q, retriever, k, mode)) for q in qs}
    name = f"final_{tag(model)}_k{k}_{mode}_{split}" + (f"_s{n}" if n > 1 else "")
    t0 = time.time()
    out = client.run(jobs, f"{OUT}/{name}.jsonl", n=n, temperature=temperature, max_tokens=400 if mode == "direct" else 700)
    print(f"{name}: {(time.time() - t0) / max(1, len(qs)):.1f} s/question (0 if loaded from runs/)")
    return out

CANDIDATES = ["qwen2.5:32b", "qwen3:32b", "gemma3:27b"]   # more if time allows: "mistral-small3.2:24b", "phi4:14b"
for m in CANDIDATES:
    try:
        RAW[(m, 4, "direct")] = raw = run(m, eval_qs, docs_dev, f"dev{N}")
        RESULTS[(m, 4, "direct")] = r = score(eval_qs, solve.predictions(eval_qs, docs_dev, raw))
        print_report(r, f"{m} + example RAG (k=4)")
    except Exception as e:
        print(m, "failed:", e)
if not RESULTS:
    raise RuntimeError(f"Every model failed. Check that the server cell ran, and read {BASE}/ollama.log")
BEST = max((key for key in RESULTS if isinstance(key, tuple) and key[1:] == (4, "direct")), key=lambda key: RESULTS[key]["EM"])[0]
print("\nMost accurate model on dev:", BEST)
for key, r in sorted(RESULTS.items(), key=lambda kv: -kv[1]["EM"]):
    print(f"  {r['EM']:6.2f} EM  {r['F1']:6.2f} F1  {key}")'''),
    md("## Part B · What does each part add? (ablation, same 200 dev questions)\n"
       "No RAG (k=0) vs. 4 or 8 retrieved examples, and answering directly vs. a short explanation first."),
    code(r'''for k, mode in ((0, "direct"), (8, "direct"), (4, "reason")):
    RAW[(BEST, k, mode)] = raw = run(BEST, eval_qs, docs_dev, f"dev{N}", k, mode)
    RESULTS[(BEST, k, mode)] = score(eval_qs, solve.predictions(eval_qs, docs_dev, raw))
print(f"{BEST}, 200 dev questions, official scorer:")
for key in [(BEST, 0, "direct"), (BEST, 4, "direct"), (BEST, 8, "direct"), (BEST, 4, "reason")]:
    r = RESULTS[key]
    print(f"  k={key[1]} {key[2]:<6}  EM {r['EM']:6.2f}  F1 {r['F1']:6.2f}  unit {r['scale_acc']:5.1f}  "
          + "  ".join(f"{t} {r.get(t + '_EM', 0):5.1f}" for t in ("span", "multi-span", "arithmetic", "count")))
K_BEST, MODE_BEST = max((key[1:] for key in RESULTS if isinstance(key, tuple) and key[0] == BEST),
                        key=lambda km: RESULTS[(BEST, *km)]["EM"])
print("best setting:", K_BEST, MODE_BEST)'''),
    md("## Part C · Ensemble with TAT-LLM-7B + voting where they disagree (200 dev questions)\n"
       "TAT-LLM's answers come from notebook 02 (no rerun). Only questions where the two systems disagree get 4 extra "
       "samples, which keeps the cost low. The two weights are chosen on dev; prefer the simplest setting within "
       "0.5 EM of the best, since 200 questions is a small sample."),
    code(r'''tat_dev = read_jsonl(f"{OUT}/tat-llm-7b-fft_dev200.jsonl")
if not tat_dev:
    print("TAT-LLM outputs not found: ensemble skipped. Voting is used where the model's answer is missing or "
          "not found on the page. Upload tat-llm-7b-fft_dev200.jsonl (notebook 02) to add the ensemble.")
greedy = RAW[(BEST, K_BEST, MODE_BEST)]

def disagreements(qs, docs, greedy, tat):
    """Questions that get extra samples: the two systems disagree (or, without TAT-LLM, the model's
    answer is missing or ungrounded)."""
    out = []
    for q in qs:
        doc = docs[q["doc_uid"]]
        a = solve.candidate(greedy[q["uid"]]["raw"], doc) if q["uid"] in greedy else None
        if not tat:
            if a is None or not a["grounded"]:
                out.append(q)
            continue
        b = solve.tatllm_candidate(tat[q["uid"]]["raw"], doc) if q["uid"] in tat else None
        if a is None or b is None or a["key"] != b["key"]:
            out.append(q)
    return out

dis = disagreements(eval_qs, docs_dev, greedy, tat_dev)
print(f"{len(dis)} of {len(eval_qs)} dev questions get 4 extra samples")
VOTE_K = 4
samples = run(BEST, dis, docs_dev, f"dev{N}", K_BEST, MODE_BEST, n=VOTE_K, temperature=0.7)

grid = {}
for tw in ((0.0, 0.5, 1.0, 1.5, 2.0) if tat_dev else (0.0,)):
    for uw in (0.25, 0.5, 1.0):
        p = solve.predictions(eval_qs, docs_dev, greedy, samples, tat_dev, tw, uw)
        grid[(tw, uw)] = score(eval_qs, p)["EM"]
for (tw, uw), em in sorted(grid.items(), key=lambda kv: -kv[1])[:8]:
    print(f"  TAT-LLM weight {tw:3}  ungrounded weight {uw:4}  EM {em:6.2f}")
best_em = max(grid.values())
TAT_W, UNG_W = min((k for k, em in grid.items() if em >= best_em - 0.5), key=lambda k: (k[0], -k[1]))
final_dev = solve.predictions(eval_qs, docs_dev, greedy, samples, tat_dev, TAT_W, UNG_W)
RESULTS["final"] = score(eval_qs, final_dev)
print_report(RESULTS["final"], f"FinSight final on dev: {BEST} k={K_BEST} {MODE_BEST} + TAT-LLM x{TAT_W} + voting")

# confidence: is the vote share a useful "I'm not sure" signal?
rows = {r["uid"]: r for r in RESULTS["final"]["rows"]}
for t in (0.5, 0.75, 1.0):
    kept = [q for q in eval_qs if final_dev[q["uid"]]["agreement"] >= t]
    em = 100 * sum(rows[q["uid"]]["em"] for q in kept) / max(1, len(kept))
    print(f"  answer only when agreement >= {t}: {100 * len(kept) / len(eval_qs):5.1f}% answered, EM {em:5.1f}")
json.dump({"best_model": BEST, "k": K_BEST, "mode": MODE_BEST, "tatllm_weight": TAT_W, "ungrounded_weight": UNG_W,
           "results": {str(k): report_row(r) for k, r in RESULTS.items()}},
          open(f"{OUT}/final_dev{N}_results.json", "w"), indent=1)'''),
    md("## Part D · Risk analysis\n"
       "1. Accuracy on the risk-related dev questions (debt, liabilities, equity, margins, interest, operating cash flow).\n"
       "2. The automatic risk report on documents that contain balance-sheet totals."),
    code(r'''from IPython.display import HTML, display
from finsight import risk
sub = risk.risk_questions(eval_qs)
print_report(score(sub, final_dev), f"risk-related dev questions ({len(sub)})")

client = use_model(BEST)
with_totals = [d for d in docs_dev.values()
               if sum(w.lower() == "current" for p in d["pages"] for b in p["blocks"] for w in b["words"]) >= 2][:3]
for doc in with_totals:
    rep = risk.report(client, doc)
    display(HTML(risk.render_html(rep)))'''),
    md("## Part E · Final run on the official TEST set (run once, at the end)\n"
       "**Off by default** so *Run all* is safe. Uses the model, setting and weights chosen on dev above; nothing is "
       "tuned on test. Needs `tat-llm-7b-fft_test.jsonl` from notebook 02. Writes `final_test_official.json` "
       "(`{uid: [answer, scale]}`, the leaderboard format) and `final_test_results.json`."),
    code(r'''RUN_TEST = False
if not RUN_TEST:
    raise SystemExit("Part E skipped (RUN_TEST = False).")
from finsight import risk
tat_test =read_jsonl(f"{OUT}/tat-llm-7b-fft_test.jsonl") if TAT_W else {}
assert not TAT_W or len(tat_test) == len(q_test), "Upload tat-llm-7b-fft_test.jsonl (all 1,663 questions) from notebook 02"
greedy_t = run(BEST, q_test, docs_test, "test", K_BEST, MODE_BEST)
dis_t = disagreements(q_test, docs_test, greedy_t, tat_test)
print(len(dis_t), "test questions get extra samples")
samples_t = run(BEST, dis_t, docs_test, "test", K_BEST, MODE_BEST, n=VOTE_K, temperature=0.7)

FINAL = f"FinSight final ({'+ TAT-LLM x' + str(TAT_W) + ' ' if TAT_W else ''}+ voting)"
TEST = {f"{BEST} + example RAG + executor": solve.predictions(q_test, docs_test, greedy_t)}
if tat_test:
    TEST["TAT-LLM-7B + executor (notebook 02)"] = solve.predictions(q_test, docs_test, {}, None, tat_test, 1.0)
TEST[FINAL] = solve.predictions(q_test, docs_test, greedy_t, samples_t, tat_test, TAT_W, UNG_W)
SCORES = {label: score(q_test, p) for label, p in TEST.items()}
for label, r in SCORES.items():
    print_report(r, "TEST: " + label)
final_t = TEST[FINAL]
sub_t = risk.risk_questions(q_test)
print_report(score(sub_t, final_t), f"TEST risk-related questions ({len(sub_t)})")
print("\nPublished TAT-DQA test results (official leaderboard, EM / F1): MHST 41.5 / 50.7 · Doc2SoarGraph 59.2 / 67.6 · "
      "TAT-LLM-7B 71.3 / 80.2 · TAT-LLM-13B 72.2 / 80.5 · TAT-LLM-70B 76.5 / 83.9 · Human 84.1 / 90.8")
json.dump({u: [p["answer"], p["scale"]] for u, p in final_t.items()}, open(f"{OUT}/final_test_official.json", "w"))
json.dump({"setting": {"model": BEST, "k": K_BEST, "mode": MODE_BEST, "tatllm_weight": TAT_W, "ungrounded_weight": UNG_W},
           "results": {k: report_row(r) for k, r in SCORES.items()},
           "risk_subset": report_row(score(sub_t, final_t))}, open(f"{OUT}/final_test_results.json", "w"), indent=1)'''),
    md("## Part F · Demo app: ask a question, or get the automatic risk report\n"
       "Upload a digital financial PDF (up to 3 pages). The live app uses the main model with example RAG (TAT-LLM "
       "is not loaded here, so there is no ensemble)."),
    code(r'''import gradio as gr
from finsight.demo import doc_from_pdf, highlight_pages, explain_final_html
client = use_model(BEST)

def ask(pdf, question):
    if pdf is None or not question.strip():
        return "<p>Upload a PDF and type a question.</p>", []
    path = pdf if isinstance(pdf, str) else pdf.name
    doc, scales = doc_from_pdf(path)
    res = solve.answer(client, doc, question, retriever, K_BEST, n_samples=2, mode=MODE_BEST)
    return explain_final_html(res, doc), highlight_pages(path, doc, scales, res["evidence"])

def risk_report(pdf):
    if pdf is None:
        return "<p>Upload a PDF.</p>", []
    path = pdf if isinstance(pdf, str) else pdf.name
    doc, scales = doc_from_pdf(path)
    rep = risk.report(client, doc)
    rows = sorted({r for c in rep["checks"] for r in c["evidence"]} | {u["row"] for u in rep["unusual_changes"]})
    return risk.render_html(rep), highlight_pages(path, doc, scales, rows)

with gr.Blocks(title="FinSight") as app:
    gr.Markdown("## FinSight: answers from financial statements, with the calculation and the source rows")
    pdf = gr.File(label="Financial PDF (digital, up to 3 pages)", file_types=[".pdf"])
    with gr.Tab("Ask a question"):
        q = gr.Textbox(label="Question", placeholder="What was the percentage change in revenue from 2018 to 2019?")
        b1 = gr.Button("Answer")
        h1, g1 = gr.HTML(), gr.Gallery(label="Evidence on the page", columns=1, height=800)
        b1.click(ask, [pdf, q], [h1, g1])
    with gr.Tab("Automatic risk report"):
        b2 = gr.Button("Analyse risk")
        h2, g2 = gr.HTML(), gr.Gallery(label="Rows used", columns=1, height=800)
        b2.click(risk_report, [pdf], [h2, g2])
app.launch(share=True)'''),
    code(r'''# Kaggle: download all results in one file (Output panel -> runs.zip), then upload them to Drive / send them
shutil.make_archive(f"{BASE}/runs", "zip", OUT)
print("saved", f"{BASE}/runs.zip")'''),
])

nb_dir = os.path.join(ROOT, "notebooks")
os.makedirs(nb_dir, exist_ok=True)
for old in glob.glob(os.path.join(nb_dir, "*.ipynb")):
    os.remove(old)
for name, nb in (("01_baseline_general_llm.ipynb", baseline), ("02_finsight_pipeline.ipynb", pipeline),
                 ("03_demo_app.ipynb", demo), ("04_paper1_vlm_rerun.ipynb", paper1),
                 ("05_final_pipeline.ipynb", final)):
    with open(os.path.join(nb_dir, name), "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("wrote notebooks/" + name)
