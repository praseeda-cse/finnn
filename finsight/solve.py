"""FinSight final pipeline. No training anywhere.

    page text ─► example RAG (similar solved training questions) ─► open LLM writes
    evidence · type · program/span · scale ─► Python executes the program ─► grounding check ─►
    vote over samples and over systems (optionally with TAT-LLM) ─► answer + evidence + confidence

The model is reached through llm.ChatClient, so any open model served by Ollama, vLLM or llama.cpp
can be plugged in and compared on dev before the single test run.
"""
import re
from collections import Counter

from . import tatllm
from .data import render_doc
from .prompt import build_messages, parse_output, to_prediction
from .tatqa_utils import normalize_answer

SYSTEM = """You answer questions about a financial document. The document is given as numbered rows ([id] text); table cells are separated by " | ".
{mode}
End your reply with exactly these four lines:
evidence: <comma-separated row ids you used>
type: <span | multi-span | arithmetic | count>
<one of:
  answer: <JSON list of text copied exactly from the document>   (span: one item, multi-span: several)
  program: <arithmetic expression using numbers copied from the document>   (arithmetic; it will be executed)
  items: <JSON list of the things being counted>   (count)>
scale: <empty | thousand | million | billion | percent>
Rules:
- Never compute a number yourself. For arithmetic write the program; Python executes it.
- The change from an earlier value A to a later value B is B - A. A percentage change is (B - A) / A * 100 with scale percent. An average of n values is (v1 + ... + vn) / n.
- In a program, a value printed in parentheses, such as (1,234), is negative: write -1234.
- If the answer itself is printed in the document, copy it as a span instead of computing it.
- The scale is the unit of the answer: values from a table "in thousands" have scale thousand, percentages have scale percent, counts and plain ratios have an empty scale.
- The solved examples come from the same kind of documents: follow their conventions."""

MODES = {
    "direct": "Reply with only the four lines, nothing else.",
    "reason": "First explain in at most three short sentences which rows and which calculation are needed. Then write the four lines.",
}
SMALL_CONSTANTS = {"1", "2", "3", "4", "5", "10", "12", "100", "1000"}
NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
LAST_BLOCK = re.compile(r"^\s*\**evidence\**\s*:", re.I | re.M)


def messages(doc, question, retriever=None, k=4, mode="direct"):
    """Chat messages for one question: system rules, k retrieved solved examples, the document."""
    examples = retriever(question, k, doc=doc) if retriever is not None and k else []
    return build_messages(render_doc(doc), question["question"], examples=examples,
                          system=SYSTEM.format(mode=MODES[mode]))


def final_block(text):
    """The four answer lines, after any reasoning, markdown bold or code fences."""
    text = text.replace("```", "").replace("**", "")
    starts = [m.start() for m in LAST_BLOCK.finditer(text)]
    return text[starts[-1]:] if starts else text


def _doc_index(doc):
    words = [w for p in doc["pages"] for b in p["blocks"] for w in b["words"]]
    numbers = {n.replace(",", "").rstrip(".") for w in words for n in NUM.findall(w)}
    return numbers, "".join(words).lower(), {b["id"] for p in doc["pages"] for b in p["blocks"]}


def grounded(doc, parsed, index=None):
    """True when every operand of the program (other than small constants) and every copied span
    occurs in the document: an answer built from numbers that are not on the page is suspect."""
    numbers, text, _ = index or _doc_index(doc)
    if parsed["program"]:
        ops = {n.replace(",", "").rstrip(".") for n in NUM.findall(parsed["program"])}
        return all(o in numbers for o in ops - SMALL_CONSTANTS)
    if parsed["answer"]:
        return all("".join(a.lower().split()) in text for a in parsed["answer"])   # spacing differs: "55%" vs "55 %"
    return True


def key(answer, scale):
    return tuple(sorted(normalize_answer(str(a)) for a in answer)), scale


def candidate(raw, doc, index=None):
    """One model reply -> executed candidate answer, or None if nothing usable was produced."""
    parsed = parse_output(final_block(raw))
    answer, scale = to_prediction(parsed)
    if not answer:
        return None
    index = index or _doc_index(doc)
    return {"key": key(answer, scale), "answer": answer, "scale": scale, "type": parsed["type"],
            "evidence": [i for i in parsed["evidence"] if i in index[2]],
            "grounded": grounded(doc, parsed, index), "raw": raw}


def tatllm_candidate(raw, doc):
    """A TAT-LLM reply (its own 5-step format, executed) as a candidate for the ensemble."""
    p = tatllm.parse(raw)
    answer, scale = tatllm.to_prediction(p, True)
    if not answer:
        return None
    evidence, found = tatllm.ground(doc, p)
    return {"key": key(answer, scale), "answer": answer, "scale": scale, "type": p["type"],
            "evidence": evidence, "grounded": found, "raw": raw}


def combine(groups, ungrounded_weight=0.5):
    """Weighted vote. groups: [(weight, [candidates...]), ...]; the first candidate of the first group
    (the main model's greedy answer) wins ties. Returns the winner plus its share of the vote."""
    votes, first = Counter(), None
    for weight, cands in groups:
        for c in cands:
            if c is None:
                continue
            first = first or c
            votes[c["key"]] += weight * (1.0 if c["grounded"] else ungrounded_weight)
    if not votes:
        return {"answer": [], "scale": "", "evidence": [], "type": "", "grounded": False, "raw": "", "agreement": 0.0}
    best = max(votes, key=lambda k: (votes[k], k == first["key"]))
    winner = next(c for _, cands in groups for c in cands if c is not None and c["key"] == best)
    return {**winner, "agreement": votes[best] / sum(votes.values())}


def predictions(questions, docs, greedy, samples=None, tatllm_raw=None, tatllm_weight=1.0, ungrounded_weight=0.5):
    """Final predictions for evaluate.score.
    greedy: {uid: {"raw"}} from the main model; samples: {uid: {"samples": [...]}} for voting;
    tatllm_raw: {uid: {"raw"}} from TAT-LLM, joined to the vote with tatllm_weight."""
    out = {}
    for q in questions:
        doc, uid = docs[q["doc_uid"]], q["uid"]
        index = _doc_index(doc)
        main = [candidate(greedy[uid]["raw"], doc, index)] if uid in greedy else []
        if samples and uid in samples:
            main += [candidate(r, doc, index) for r in samples[uid]["samples"]]
        groups = [(1.0, main)]
        if tatllm_raw and uid in tatllm_raw and tatllm_weight:
            groups.append((tatllm_weight, [tatllm_candidate(tatllm_raw[uid]["raw"], doc)]))
        out[uid] = combine(groups, ungrounded_weight)
    return out


def answer(client, doc, question, retriever=None, k=4, n_samples=0, mode="direct", max_tokens=512):
    """Answer one question live (demo, risk report). question: text or {"question": ...}."""
    q = question if isinstance(question, dict) else {"question": question}
    msgs = messages(doc, q, retriever, k, mode)
    index = _doc_index(doc)
    cands = [candidate(client.chat(msgs, 0.0, max_tokens), doc, index)]
    cands += [candidate(client.chat(msgs, 0.7, max_tokens, seed=i), doc, index) for i in range(n_samples)]
    return combine([(1.0, cands)])
