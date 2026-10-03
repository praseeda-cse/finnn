"""Tiny end-to-end GPU run (train a few steps, predict, score) to catch bugs before
spending Colab hours. Uses Qwen2.5-0.5B so it fits a 6 GB laptop GPU.

    python scripts/smoke_test.py
"""
import os
import shutil
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from finsight.data import load_split
from finsight.evaluate import print_report, score
from finsight.infer import load_model, predict
from finsight.train import train

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
OUT = os.path.join(ROOT, "runs", "smoke")
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT)

docs_tr, q_tr = load_split(os.path.join(ROOT, "data", "tatdqa_train.json.gz"))
docs_dev, q_dev = load_split(os.path.join(ROOT, "data", "tatdqa_dev.json.gz"))
q_dev = q_dev[:24]

model, tok = load_model(MODEL)
preds = predict(model, tok, docs_dev, q_dev, os.path.join(OUT, "zeroshot.jsonl"), few_shot=True, batch_size=4)
print_report(score(q_dev, preds), "0.5B few-shot (smoke)")
print("sample raw output:\n", next(iter(preds.values()))["raw"])
del model

adapter = train(MODEL, docs_tr, q_tr, os.path.join(OUT, "train"), max_examples=64, batch_size=1, grad_accum=4,
                save_steps=10)
model, tok = load_model(MODEL, adapter=adapter)
preds = predict(model, tok, docs_dev, q_dev, os.path.join(OUT, "finetuned.jsonl"), batch_size=4)
print_report(score(q_dev, preds), "0.5B after 16 steps (smoke)")
print("sample raw output:\n", next(iter(preds.values()))["raw"])
