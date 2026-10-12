"""Batched generation with resume: predictions are appended to a .jsonl file as they
are produced, so a Colab disconnect only loses the current batch."""
import json
import os

import torch

from .data import render_doc
from .prompt import build_messages, parse_output, to_prediction


def compute_dtype():
    # bf16 only where it is native (Ampere+: L4, A100). A Colab T4 would emulate it slowly.
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        return torch.bfloat16
    return torch.float16


def load_model(model_id, adapter=None, four_bit=True):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kwargs = {"device_map": "auto", "dtype": compute_dtype()}
    if four_bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype(), bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tok


def _read_done(path):
    done = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                done[row["uid"]] = row
    return done


def _prompts(tok, docs, questions, few_shot=False, examples_fn=None, template_kwargs=None):
    """examples_fn(question) -> list of solved examples (see examples.py); template_kwargs are passed
    to the chat template, e.g. {"enable_thinking": False} for Qwen3."""
    rendered, prompts = {}, []
    for q in questions:
        if q["doc_uid"] not in rendered:
            rendered[q["doc_uid"]] = render_doc(docs[q["doc_uid"]])
        msgs = build_messages(rendered[q["doc_uid"]], q["question"], few_shot=few_shot,
                              examples=examples_fn(q) if examples_fn else None)
        prompts.append((q["uid"], tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                          **(template_kwargs or {}))))
    prompts.sort(key=lambda p: len(p[1]))  # similar lengths per batch = less padding
    return prompts


@torch.no_grad()
def sample(model, tok, docs, questions, out_path, k=5, temperature=0.7, top_p=0.95, batch_size=2,
           max_new_tokens=160, examples_fn=None, template_kwargs=None):
    """k sampled outputs per question, for voting (Step 5.6) and expert iteration (Step 4.5).
    Returns {uid: [raw text, ...]}; resumable like predict()."""
    done = {uid: row["samples"] for uid, row in _read_done(out_path).items()}
    prompts = _prompts(tok, docs, [q for q in questions if q["uid"] not in done], False, examples_fn, template_kwargs)
    with open(out_path, "a", encoding="utf-8") as f:
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            enc = tok([p for _, p in batch], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature,
                                 top_p=top_p, num_return_sequences=k, pad_token_id=tok.pad_token_id)
            texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for j, (uid, _) in enumerate(batch):
                done[uid] = texts[j * k:(j + 1) * k]
                f.write(json.dumps({"uid": uid, "samples": done[uid]}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"\r{len(done)}/{len(questions)} sampled (k={k})", end="")
    print()
    return done


@torch.no_grad()
def predict(model, tok, docs, questions, out_path, few_shot=False, batch_size=8, max_new_tokens=160,
            examples_fn=None, template_kwargs=None):
    """Returns {uid: {"answer", "scale", "evidence", "type", "raw"}} for all questions."""
    done = _read_done(out_path)
    prompts = _prompts(tok, docs, [q for q in questions if q["uid"] not in done], few_shot, examples_fn,
                       template_kwargs)

    with open(out_path, "a", encoding="utf-8") as f:
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            enc = tok([p for _, p in batch], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
            texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for (uid, _), text in zip(batch, texts):
                parsed = parse_output(text)
                answer, scale = to_prediction(parsed)
                row = {"uid": uid, "answer": answer, "scale": scale, "evidence": parsed["evidence"],
                       "type": parsed["type"], "raw": text}
                done[uid] = row
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(f"\r{len(done)}/{len(questions)} predicted", end="")
    print()
    return done
