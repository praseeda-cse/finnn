"""Prompt, training target and output parsing.

The model never writes a computed number. It answers in four lines:

    evidence: 7, 30          <- block ids it used (checked against gold block_mapping)
    type: arithmetic         <- span | multi-span | arithmetic | count
    program: (55 + 50) / 2   <- arithmetic only; Python computes it
    scale: percent           <- "", thousand, million, billion, percent

Span and multi-span use `answer: ["...", ...]` instead of `program:`;
count uses `items: [...]` and the answer is len(items).
"""
import json
import re

from .program import execute, gold_program

SCALES = ("", "thousand", "million", "billion", "percent")
TYPES = ("span", "multi-span", "arithmetic", "count")

SYSTEM = """You answer questions about a financial document. The document is given as numbered text blocks; table cells are separated by " | ".
Reply with exactly four lines and nothing else:
evidence: <comma-separated block ids you used>
type: <span | multi-span | arithmetic | count>
<one of:
  answer: <JSON list with the exact text copied from the document>   (span: one item, multi-span: several)
  program: <arithmetic expression using numbers copied from the document>   (arithmetic; it will be executed)
  items: <JSON list of the things being counted>   (count)>
scale: <empty | thousand | million | billion | percent>
Rules: never compute numbers yourself, write the program instead. For a percentage change write (new - old) / old * 100 with scale percent. The scale is the unit the answer is expressed in (e.g. "in thousands" tables give scale thousand)."""

FEW_SHOT = [
    ("[0] (in thousands) | 2019 | 2018\n[1] Revenue | $ 10,310 | $ 12,629\n[2] Cost of revenue | 4,405 | 6,295",
     "What was the percentage change in revenue from 2018 to 2019?",
     "evidence: 0, 1\ntype: arithmetic\nprogram: (10310 - 12629) / 12629 * 100\nscale: percent"),
    ("[0] (in thousands) | 2019 | 2018\n[1] Revenue | $ 10,310 | $ 12,629\n[2] Cost of revenue | 4,405 | 6,295",
     "What was the cost of revenue in 2019?",
     "evidence: 0, 2\ntype: span\nanswer: [\"4,405\"]\nscale: thousand"),
]


def user_message(context, question):
    return f"Document:\n{context}\n\nQuestion: {question}"


def build_messages(context, question, few_shot=False):
    msgs = [{"role": "system", "content": SYSTEM}]
    if few_shot:
        for ctx, q, a in FEW_SHOT:
            msgs += [{"role": "user", "content": user_message(ctx, q)}, {"role": "assistant", "content": a}]
    msgs.append({"role": "user", "content": user_message(context, question)})
    return msgs


def gold_target(q):
    """Training target for a gold question, or None if it can't be expressed."""
    t = q["answer_type"]
    lines = ["evidence: " + ", ".join(map(str, q["evidence"])), f"type: {t}"]
    if t in ("span", "multi-span"):
        ans = q["answer"] if isinstance(q["answer"], list) else [str(q["answer"])]
        lines.append("answer: " + json.dumps(ans, ensure_ascii=False))
    elif t == "arithmetic":
        prog = gold_program(q)
        if prog is None:
            return None
        lines.append(f"program: {prog}")
    elif t == "count":
        items = [s.strip() for s in q["derivation"].split("##") if s.strip()]
        if len(items) != int(q["answer"]):
            return None
        lines.append("items: " + json.dumps(items, ensure_ascii=False))
    else:
        return None
    lines.append(f"scale: {q['scale']}")
    return "\n".join(lines)


def _json_list(text):
    text = text.strip()
    try:
        val = json.loads(text)
        return [str(v) for v in val] if isinstance(val, list) else [str(val)]
    except Exception:
        text = text.strip("[]")
        return [s.strip().strip('"\'') for s in re.split(r'"\s*,\s*"', text) if s.strip()] or []


def parse_output(text):
    """Parse the model reply into a dict; tolerant of extra text and missing lines."""
    out = {"evidence": [], "type": "", "answer": None, "program": None, "items": None, "scale": ""}
    for line in text.strip().splitlines():
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        key, val = key.strip().lower(), val.strip()
        if key == "evidence":
            out["evidence"] = [int(x) for x in re.findall(r"\d+", val)]
        elif key == "type":
            out["type"] = val.lower()
        elif key == "answer":
            out["answer"] = _json_list(val)
        elif key == "program":
            out["program"] = val
        elif key == "items":
            out["items"] = _json_list(val)
        elif key == "scale":
            val = val.lower().strip(' "\'')
            out["scale"] = val if val in SCALES else ("" if val in ("empty", "none", "-") else val)
    return out


def to_prediction(parsed):
    """Return (answer_list, scale) in the format the official TAT-QA scorer expects."""
    if parsed["program"] is not None:
        value = execute(parsed["program"])
        answer = [] if value is None else [str(round(value, 4))]
    elif parsed["items"] is not None:
        answer = [str(len(parsed["items"]))]
    elif parsed["answer"] is not None:
        answer = parsed["answer"]
    else:
        answer = []
    scale = parsed["scale"] if parsed["scale"] in SCALES else ""
    return answer, scale
