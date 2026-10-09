"""QLoRA supervised fine-tuning: (document + question) -> evidence/type/program/scale.
Loss is computed only on the answer lines. Checkpoints go to `out_dir` and training
resumes from the latest one automatically, so it survives Colab disconnects."""
import glob
import os
import random

import torch

from .data import render_doc
from .infer import compute_dtype
from .prompt import build_messages, gold_target

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def build_examples(docs, questions, tok, max_len=4096, targets=None):
    """targets: optional {uid: target text} that replaces the gold target (used by expert iteration)."""
    examples, skipped = [], 0
    rendered = {}
    for q in questions:
        target = targets.get(q["uid"]) if targets is not None else gold_target(q)
        if target is None:
            skipped += 1
            continue
        if q["doc_uid"] not in rendered:
            rendered[q["doc_uid"]] = render_doc(docs[q["doc_uid"]])
        msgs = build_messages(rendered[q["doc_uid"]], q["question"])
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        full = tok.apply_chat_template(msgs + [{"role": "assistant", "content": target}], tokenize=False)
        assert full.startswith(prompt), "chat template does not extend the prompt; check the model's template"
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        t_ids = tok(full[len(prompt):], add_special_tokens=False)["input_ids"]
        if len(p_ids) + len(t_ids) > max_len:
            skipped += 1
            continue
        examples.append({"input_ids": p_ids + t_ids, "labels": [-100] * len(p_ids) + t_ids})
    print(f"{len(examples)} training examples ({skipped} skipped)")
    return examples


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        if len(batch) == 1:
            # Loss is only on the answer at the end, so only compute vocabulary logits there:
            # the model's built-in shift aligns logits[-k:] with labels[-k:].
            x = batch[0]
            k = sum(1 for t in x["labels"] if t != -100) + 1
            return {"input_ids": torch.tensor([x["input_ids"]]), "labels": torch.tensor([x["labels"][-k:]]),
                    "attention_mask": torch.ones(1, len(x["input_ids"]), dtype=torch.long), "logits_to_keep": k}
        n = max(len(x["input_ids"]) for x in batch)
        ids = [x["input_ids"] + [self.pad_id] * (n - len(x["input_ids"])) for x in batch]
        labels = [x["labels"] + [-100] * (n - len(x["labels"])) for x in batch]
        mask = [[1] * len(x["input_ids"]) + [0] * (n - len(x["input_ids"])) for x in batch]
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels), "attention_mask": torch.tensor(mask)}


def train(model_id, docs, questions, out_dir, epochs=1, lr=2e-4, lora_r=16, batch_size=1, grad_accum=16,
          max_len=4096, max_examples=None, save_steps=200, seed=0, four_bit=True, targets=None,
          init_adapter=None):
    """four_bit=True is QLoRA (needed for 7B on a 16 GB GPU). four_bit=False is plain LoRA on
    16-bit weights: faster, and fits a 3B model on a T4.
    targets / init_adapter: continue training an existing adapter on custom targets (expert iteration)."""
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    examples = build_examples(docs, questions, tok, max_len, targets)
    random.Random(seed).shuffle(examples)
    if max_examples:
        examples = examples[:max_examples]

    dtype = compute_dtype()
    if four_bit:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", dtype=dtype,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                   bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", dtype=dtype)
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    if init_adapter:
        model = PeftModel.from_pretrained(model, init_adapter, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.05,
                                                 target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    steps = max(1, len(examples) * epochs // (batch_size * grad_accum))
    args = TrainingArguments(
        output_dir=out_dir, num_train_epochs=epochs, per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum, learning_rate=lr, lr_scheduler_type="cosine",
        warmup_steps=max(1, steps // 20), logging_steps=10, save_steps=save_steps, save_total_limit=2,
        bf16=dtype == torch.bfloat16, fp16=dtype == torch.float16,
        gradient_checkpointing=True, optim="paged_adamw_8bit" if four_bit else "adamw_torch", report_to="none",
        remove_unused_columns=False, seed=seed)
    trainer = Trainer(model=model, args=args, train_dataset=examples, data_collator=Collator(tok.pad_token_id))
    resume = bool(glob.glob(os.path.join(out_dir, "checkpoint-*")))
    trainer.train(resume_from_checkpoint=resume or None)
    final = os.path.join(out_dir, "final")
    model.save_pretrained(final)
    tok.save_pretrained(final)
    print("saved adapter to", final)
    return final
