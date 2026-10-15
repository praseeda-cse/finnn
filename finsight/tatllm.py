"""Stage 2 with TAT-LLM (Zhu et al., 2024): the authors' LLaMA-2 model already fine-tuned on
FinQA, TAT-QA and TAT-DQA (next-tat/tat-llm-7b-fft, 13b-fft). No training on our side.

The prompt is rebuilt exactly as in the authors' released TAT-DQA files (checked: identical for
all 1,644 dev questions), so it also works on documents read by OCR from page images. The model
answers in a 5-step table (type, evidence, equation, answer, scale); our safe executor computes
the equation instead of trusting the model's own arithmetic, and the official scorer grades it."""
import json
import os
import re

import torch

from .program import execute

HEADER = open(os.path.join(os.path.dirname(__file__), "tatllm_header.txt"), encoding="utf-8").read()
SYSTEM = "You are a financial assistant. Always provide accurate and reliable information to the best of your abilities"
MAX_LEN, MIN_RESPONSE = 4096, 500
TYPE_MAP = {"single span": "span", "multiple spans": "multi-span", "count": "count", "arithmetic": "arithmetic"}


def doc_text(doc):
    """'Page k: ' + all words of the page joined by spaces, pages separated by newlines."""
    return "\n".join(f"Page {i}: " + " ".join(" ".join(b["words"]) for b in p["blocks"])
                     for i, p in enumerate(doc["pages"], 1))


def user_prompt(doc_str, question):
    return f"{HEADER}{doc_str}\n\n\n\n\n### Question \n\n{question}\n\n\n\n### Response"


def wrap(text):
    """LLaMA-2 chat format the model was trained with (the tokenizer adds <s>)."""
    return f"[INST] <<SYS>>\n{SYSTEM}\n<</SYS>>\n\n{text} [/INST]"


def build_input(tok, doc, question):
    """Full model input; the document is cut (from the end) if the prompt would leave fewer than
    500 tokens for the answer, as in the authors' inference script."""
    d = doc_text(doc)
    fixed = len(tok(wrap(user_prompt("", question)), add_special_tokens=True)["input_ids"])
    budget = MAX_LEN - MIN_RESPONSE - fixed
    ids = tok(d, add_special_tokens=False)["input_ids"]
    if len(ids) > budget:
        d = tok.decode(ids[:max(0, budget)])
    return wrap(user_prompt(d, question))


def parse(text):
    """Read the 5-step table and the final 'The answer is: ... #### ... scale is: ...' line."""
    rows = {}
    for line in text.splitlines():
        cells = [c.strip() for c in line.split("|") if c.strip()]
        if len(cells) >= 2 and cells[0] in "12345":
            rows[cells[0]] = "|".join(cells[1:]).strip()
    final_answer, final_scale = None, None
    m = re.search(r"the answer is:\s*(.*?)\s*####\s*and its corresponding scale is:\s*([a-z]+)", text, re.I | re.S)
    if m:
        final_answer, final_scale = m.group(1).strip(), m.group(2).strip().lower()
    qtype = TYPE_MAP.get(rows.get("1", "").lower(), "")
    scale = (rows.get("5") or final_scale or "none").lower().strip()
    return {"type": qtype, "evidence_text": [e.strip() for e in rows.get("2", "").split("#") if e.strip()],
            "equation": rows.get("3", ""), "answer_text": rows.get("4") or final_answer or "",
            "scale": "" if scale in ("none", "n.a.", "") else scale}


def _clean_equation(eq):
    """Same idea as the authors' clean_equation: drop letters, $, % and thousands commas."""
    eq = re.sub(r"(?<=\d),(?=\d{3})", "", eq)
    return re.sub(r"[a-zA-Z$%&]", "", eq).strip()


def _close(a, b):
    return abs(a - b) <= max(0.011, 0.002 * abs(b))


def _reconcile(value, answer_text, scale, equation):
    """Gold-style equations omit the x100 of percentages ((new-old)/old for a % change). Use the
    model's stated answer and the predicted scale to put the exact executed value on the right footing."""
    from .tatqa_utils import to_number
    stated = to_number(answer_text) if answer_text else None
    if stated is not None:
        if _close(value, stated):
            return value
        if _close(value * 100, stated):
            return value * 100
    if scale == "percent" and "100" not in equation and "/" in equation and abs(value) < 10:
        return value * 100
    return value


def to_prediction(parsed, use_executor=True):
    """(answer list, scale) for the official scorer. With use_executor, arithmetic answers come from
    executing the model's equation with our safe evaluator; otherwise from the model's own answer."""
    t, ans = parsed["type"], parsed["answer_text"]
    if t == "arithmetic" and use_executor:
        eq = _clean_equation(parsed["equation"])
        value = execute(eq)
        if value is not None:
            return [str(round(_reconcile(value, ans, parsed["scale"], eq), 4))], parsed["scale"]
    if t == "multi-span":
        return [a.strip() for a in ans.split("#") if a.strip()], parsed["scale"]
    if t == "count" and not re.fullmatch(r"\s*\d+\s*", ans or ""):
        return [str(len(parsed["evidence_text"]))], parsed["scale"]
    return ([ans] if ans else []), parsed["scale"]


def load(model_id="next-tat/tat-llm-7b-fft", four_bit=True):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from .infer import compute_dtype
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    tok.pad_token = tok.pad_token or tok.unk_token
    kwargs = {"device_map": "auto", "dtype": compute_dtype()}
    if four_bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_compute_dtype=compute_dtype(),
                                                           bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    return model, tok


def _read(out_path):
    done = {}
    if os.path.exists(out_path):
        for line in open(out_path, encoding="utf-8"):
            row = json.loads(line)
            done[row["uid"]] = row
    return done


@torch.no_grad()
def _generate_all(model, tok, docs, questions, out_path, batch_size, max_new_tokens, k=1, temperature=None):
    """Shared, resumable generation. k = 1 with temperature None: greedy, row {"uid", "raw"}.
    Otherwise k samples per question, row {"uid", "samples": [...]}. Long batches that run out of
    GPU memory are retried in halves."""
    done = _read(out_path)
    todo = [(q["uid"], build_input(tok, docs[q["doc_uid"]], q["question"])) for q in questions if q["uid"] not in done]
    todo.sort(key=lambda x: len(x[1]))
    gen_kwargs = {"max_new_tokens": max_new_tokens, "pad_token_id": tok.pad_token_id}
    if temperature is None:
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.95, num_return_sequences=k)

    def generate(batch):
        try:
            enc = tok([p for _, p in batch], return_tensors="pt", padding=True).to(model.device)
            gen = model.generate(**enc, **gen_kwargs)
            return tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(batch) == 1:
                raise
            half = len(batch) // 2
            return generate(batch[:half]) + generate(batch[half:])

    per = 1 if temperature is None else k
    with open(out_path, "a", encoding="utf-8") as f:
        for i in range(0, len(todo), batch_size):
            batch = todo[i:i + batch_size]
            texts = generate(batch)
            for j, (uid, _) in enumerate(batch):
                outs = texts[j * per:(j + 1) * per]
                done[uid] = {"uid": uid, "raw": outs[0]} if temperature is None else {"uid": uid, "samples": outs}
                f.write(json.dumps(done[uid], ensure_ascii=False) + "\n")
            f.flush()
            print(f"\r{len(done)}/{len(questions)} done", end="")
    print()
    return done


def predict(model, tok, docs, questions, out_path, batch_size=4, max_new_tokens=400):
    """Greedy answers: {uid: {"uid", "raw"}}. Scored separately, with or without the executor."""
    return _generate_all(model, tok, docs, questions, out_path, batch_size, max_new_tokens)


def sample(model, tok, docs, questions, out_path, k=5, temperature=0.7, batch_size=2, max_new_tokens=400):
    """k sampled answers per question for self-consistency voting: {uid: {"uid", "samples"}}."""
    return _generate_all(model, tok, docs, questions, out_path, batch_size, max_new_tokens, k, temperature)


def score_outputs(raw_by_uid, use_executor=True, doc_of=None):
    """{uid: {"raw"}} -> predictions dict in the format of evaluate.score.
    doc_of: optional {uid: doc}; then evidence rows are grounded for evidence recall / precision."""
    preds = {}
    for uid, row in raw_by_uid.items():
        p = parse(row["raw"])
        answer, scale = to_prediction(p, use_executor)
        evidence = ground(doc_of[uid], p)[0] if doc_of else []
        preds[uid] = {"answer": answer, "scale": scale, "evidence": evidence, "type": p["type"], "raw": row["raw"]}
    return preds


def _key(answer, scale):
    from .tatqa_utils import normalize_answer
    return tuple(sorted(normalize_answer(str(a)) for a in answer)), scale


def vote(raws, greedy_raw=None, doc=None):
    """Self-consistency: execute every sampled answer and return the most common (answer, scale).
    The greedy answer, if given, joins the vote and wins ties. Also returns the agreement share,
    usable as a confidence score."""
    from collections import Counter
    candidates = ([greedy_raw] if greedy_raw else []) + list(raws)
    preds = []
    for raw in candidates:
        answer, scale = to_prediction(parse(raw), True)
        if answer:
            preds.append((_key(answer, scale), answer, scale, raw))
    if not preds:
        return {"answer": [], "scale": "", "evidence": [], "raw": candidates[0] if candidates else "", "agreement": 0.0}
    counts = Counter(k for k, *_ in preds)
    best = max(counts, key=lambda k: (counts[k], k == preds[0][0]))  # ties -> greedy's answer
    _, answer, scale, raw = next(p for p in preds if p[0] == best)
    evidence = ground(doc, parse(raw))[0] if doc else []
    return {"answer": answer, "scale": scale, "evidence": evidence, "raw": raw, "agreement": counts[best] / len(candidates)}


def ground(doc, parsed):
    """Block ids that contain the model's evidence strings (or the numbers of its equation), for
    highlighting the evidence on the page and checking that operands exist in the document."""
    blocks = [b for p in doc["pages"] for b in p["blocks"]]
    num = lambda s: s.replace(",", "").strip("$%().")
    tokens = {b["id"]: {num(w) for w in b["words"]} for b in blocks}
    texts = {b["id"]: " ".join(b["words"]).lower() for b in blocks}
    keys = []  # (kind, value): numbers match whole tokens, text matches substrings
    for e in parsed["evidence_text"]:
        e = e.strip()
        if re.fullmatch(r"[-$(]*[\d,]+\.?\d*[%)]*", e):
            keys.append(("num", num(e)))
        elif len(e) > 2:
            keys.append(("text", e.lower()))
    keys += [("num", n.rstrip(".")) for n in re.findall(r"\d[\d.]*", _clean_equation(parsed["equation"]))]
    keys = list(dict.fromkeys(k for k in keys if k[1]))
    hit = lambda bid, k: k[1] in tokens[bid] if k[0] == "num" else k[1] in texts[bid]
    found = all(any(hit(b, k) for b in tokens) for k in keys)
    cover = {b: sum(hit(b, k) for k in keys) for b in tokens}
    best = max(cover.values(), default=0)
    if best == 0:
        return [], found
    if best > 1:   # rows holding several evidence values together are the evidence
        return sorted(b for b, c in cover.items() if c == best), found
    ids = set()    # otherwise the first row holding each value
    for k in keys:
        first = next((b for b in tokens if hit(b, k)), None)
        if first is not None:
            ids.add(first)
    return sorted(ids), found
