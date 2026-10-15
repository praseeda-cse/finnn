"""Paper 1 follow-up: Qwen2-VL-2B on TAT-DQA page images, end to end.

Re-runs the original zero-shot setup with a longer output budget, with one prompt that does not use
the gold answer type, and with a compact structured output (answer or arithmetic expression + unit)
whose expression is executed by our safe evaluator. Pages are the same 150-DPI renders as the
original run (one page per document, chosen as in the original loader)."""
import json
import os
import re

import torch

from .program import execute
from .tatqa_utils import to_number

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
MIN_PIXELS, MAX_PIXELS = 256 * 28 * 28, 1536 * 28 * 28

# --- prompts ---------------------------------------------------------------------------------
# "orig": identical to the original run (fp/run_full_test_eval.py); chosen by the GOLD answer type.
ORIG_INSTRUCTIONS = {
    "arithmetic": ("1. Locate the exact values in the document.\n"
                   "2. Write out the calculation formula step by step.\n"
                   "3. State your final calculated number at the end as: Answer: <final_result>"),
    "multi-span": ("Extract all matching items/values from the document page.\n"
                   "Format your final output at the end as: Answer: <val1>, <val2>, ..."),
    "other": ("Extract the exact text or numerical span from the document.\n"
              "Format your final output at the end as: Answer: <exact_span>"),
}
# "generic": one prompt for every question (no gold information).
GENERIC_INSTRUCTIONS = ("Answer the question from the document. If a calculation is needed, show it briefly.\n"
                        "Format your final output at the end as: Answer: <answer>")
# "structured": compact output; calculations are returned as an expression and executed by Python.
STRUCTURED_INSTRUCTIONS = (
    "Reply in exactly two lines and nothing else:\n"
    "Answer: <the answer copied from the document. If the question needs a calculation, write ONLY the "
    "arithmetic expression with the numbers from the document, e.g. (958744 - 937510) / 937510 * 100. "
    "If there are several answers, separate them with ' | '>\n"
    "Unit: <none, thousand, million or percent>")


def prompt(question, mode, answer_type=None):
    if mode == "orig":
        instr = ORIG_INSTRUCTIONS.get(answer_type, ORIG_INSTRUCTIONS["other"])
    elif mode == "generic":
        instr = GENERIC_INSTRUCTIONS
    else:
        instr = STRUCTURED_INSTRUCTIONS
    return f"You are an expert financial document assistant.\nQuestion: {question}\n\n{instr}"


# --- output handling -------------------------------------------------------------------------
def clean_prediction(text):
    """The original run's cleaner (fp/postprocess_full_results.py), unchanged."""
    if not text:
        return ""
    text = re.sub(r"</?(?:exact_span|val\d+|answer|result)[^>]*>", "", text, flags=re.IGNORECASE)
    if "Answer:" in text:
        text = text.split("Answer:")[-1].strip()
    elif "answer:" in text:
        text = text.split("answer:")[-1].strip()
    elif "is:" in text and not re.search(r"\d", text.split("is:")[0]):
        text = text.split("is:")[-1].strip()
    elif "=" in text and not re.match(r"^[\d\.\,\s\-\+\*\/\%]+$", text):
        text = text.split("=")[-1].strip()
    text = re.sub(r"<val\d+>", "", text, flags=re.IGNORECASE)
    text = text.rstrip(".")
    return text.strip("'\"` \t\n")


UNITS = {"none": "", "thousand": "thousand", "thousands": "thousand", "million": "million",
         "millions": "million", "percent": "percent", "percentage": "percent", "%": "percent"}
EXPR = re.compile(r"^[\d\s.,()+\-*/$%]+$")


def parse_structured(text):
    """(answer list, scale, executed) from the two-line structured output."""
    ans = re.search(r"answer:\s*(.*)", text, re.I)
    unit = re.search(r"unit:\s*([a-z%]+)", text, re.I)
    answer = ans.group(1).strip() if ans else text.strip().splitlines()[0] if text.strip() else ""
    answer = re.split(r"\s*unit:", answer, flags=re.I)[0].strip().rstrip(".")
    scale = UNITS.get(unit.group(1).lower(), "") if unit else ""
    if answer and EXPR.match(answer) and re.search(r"\d\s*[-+*/]\s*[\d(]", answer):
        v = execute(answer)
        if v is not None:
            if scale == "percent" and "/" in answer and "100" not in answer and abs(v) < 10:
                v *= 100
            return [str(round(v, 4))], scale, True
    parts = [a.strip() for a in answer.split("|") if a.strip()]
    return parts, scale, False


def to_preds(raw_by_uid, mode, gold_scale_of=None):
    """{uid: raw text} -> predictions for evaluate.score. gold_scale_of: optional {uid: scale}
    to attach the gold unit (diagnostic only)."""
    preds = {}
    for uid, raw in raw_by_uid.items():
        if mode == "structured":
            answer, scale, executed = parse_structured(raw)
        else:
            c = clean_prediction(raw)
            answer, scale, executed = ([c] if c else []), "", False
        if gold_scale_of is not None:
            scale = gold_scale_of[uid]
        preds[uid] = {"answer": answer, "scale": scale, "evidence": [], "executed": executed, "raw": raw}
    return preds


NUMERIC = re.compile(r"^\s*[-–]?\s*\$?\s*\(?\s*[-–]?\s*\$?\s*\d[\d,]*(\.\d+)?\s*\)?\s*%?\s*"
                     r"(thousand|million|billion|percent)?\s*\.?\s*$", re.I)


def _nums_in(text):
    return {round(abs(float(m.replace(",", ""))), 4) for m in re.findall(r"\d[\d,]*\.?\d*", text)
            if m.replace(",", "").replace(".", "").isdigit()}


def behaviour(questions, raw_by_uid, preds):
    """Arithmetic behaviour: share with a numeric final answer, reaching the 'Answer:' marker, and
    mentioning all / some / none of the gold operands."""
    out = {"n": 0, "numeric_answer": 0, "answer_marker": 0, "all_operands": 0, "some_operands": 0,
           "no_operands": 0, "executed": 0}
    for q in questions:
        if q["answer_type"] != "arithmetic" or q["uid"] not in raw_by_uid:
            continue
        raw, p = raw_by_uid[q["uid"]], preds[q["uid"]]
        out["n"] += 1
        out["answer_marker"] += "answer:" in raw.lower()
        out["numeric_answer"] += bool(p["answer"]) and bool(NUMERIC.match(str(p["answer"][0])))
        out["executed"] += p["executed"]
        facts = {round(abs(f), 4) for f in (to_number(str(x).strip("$%(),")) for x in q["facts"]) if f is not None}
        hit = len(facts & _nums_in(raw))
        if facts:
            out["all_operands" if hit == len(facts) else "some_operands" if hit else "no_operands"] += 1
    n = max(1, out["n"])
    return {k: (v if k == "n" else round(100 * v / n, 1)) for k, v in out.items()}


# --- model -----------------------------------------------------------------------------------
def load():
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    processor = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    processor.tokenizer.padding_side = "left"
    model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_ID, dtype=dtype, attn_implementation="sdpa",
                                                            device_map="cuda")
    model.eval()
    return model, processor


@torch.no_grad()
def run(model, processor, questions, image_of, mode, out_path, max_new_tokens, batch_size=4):
    """Greedy answers for one condition, resumable: {uid: raw text}. image_of: {uid: image path}."""
    from PIL import Image
    done = {}
    if os.path.exists(out_path):
        for line in open(out_path, encoding="utf-8"):
            row = json.loads(line)
            done[row["uid"]] = row["raw"]
    todo = [q for q in questions if q["uid"] not in done]
    with open(out_path, "a", encoding="utf-8") as f:
        for i in range(0, len(todo), batch_size):
            batch = todo[i:i + batch_size]
            texts, images = [], []
            for q in batch:
                msg = [{"role": "user", "content": [{"type": "image"},
                                                    {"type": "text", "text": prompt(q["question"], mode, q["answer_type"])}]}]
                texts.append(processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))
                images.append(Image.open(image_of[q["uid"]]).convert("RGB"))
            enc = processor(text=texts, images=images, return_tensors="pt", padding=True).to(model.device)
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
            outs = processor.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for q, o in zip(batch, outs):
                done[q["uid"]] = o
                f.write(json.dumps({"uid": q["uid"], "raw": o}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"\r{mode}: {len(done)}/{len(questions)}", end="")
    print()
    return {q["uid"]: done[q["uid"]] for q in questions}
