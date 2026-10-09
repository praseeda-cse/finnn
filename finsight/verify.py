"""Step 5.4: rule-based answer verification against the cited evidence.

Every check returns True / False (or None when it does not apply). `verify` also returns a
score in [0, 1] (the share of applicable checks that passed), used for weighted voting and
confidence in trust.py."""
import ast
import re

from .data import block_text
from .prompt import TYPES

# Small constants that legitimately appear in programs without being on the page
# (averages, x100 for percentages, unit conversions).
FREE_CONSTANTS = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 12.0, 100.0, 1000.0, 1e6}
PERCENT_WORDS = re.compile(r"percent|percentage|%|ratio|rate|proportion|margin|growth", re.I)
_NUM = re.compile(r"\d[\d,]*\.?\d*|\.\d+")


def _to_float(text):
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


def numbers_in(text):
    """All numbers written in `text`, as floats (accounting '(1,234)' gives both signs)."""
    values = set()
    for m in _NUM.finditer(text):
        v = _to_float(m.group(0).rstrip("."))
        if v is not None:
            values.add(v)
            values.add(-v)
    return values


def program_operands(program):
    """Numeric literals in a program (signs are handled by the comparison)."""
    from .program import clean_expression
    try:
        tree = ast.parse(clean_expression(program), mode="eval")
    except SyntaxError:
        return None
    return [float(n.value) for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, (int, float))]


def _norm(s):
    return re.sub(r"[\s$,%()]", "", s.lower())


def evidence_text(doc, ids):
    by_id = {b["id"]: b for p in doc["pages"] for b in p["blocks"]}
    return "\n".join(block_text(by_id[i]) for i in ids if i in by_id), all(i in by_id for i in ids)


def verify(parsed, question, doc):
    """parsed: output of prompt.parse_output. Returns (checks dict, score)."""
    checks = {}
    t = parsed["type"]
    checks["format"] = t in TYPES and (
        (t == "arithmetic" and parsed["program"] is not None) or
        (t in ("span", "multi-span") and bool(parsed["answer"])) or
        (t == "count" and parsed["items"] is not None))
    ev_text, ids_valid = evidence_text(doc, parsed["evidence"])
    checks["evidence_present"] = bool(parsed["evidence"]) and ids_valid

    if parsed["program"] is not None:
        ops = program_operands(parsed["program"])
        if ops is None:
            checks["operands_grounded"] = False
        else:
            page_nums = numbers_in(ev_text)
            needed = [v for v in ops if v not in FREE_CONSTANTS]
            checks["operands_grounded"] = all(any(abs(v - w) < 1e-9 for w in page_nums) for v in needed)
        is_pct_program = re.search(r"\*\s*100\b", parsed["program"]) is not None
        if is_pct_program:
            checks["scale_consistent"] = parsed["scale"] == "percent"
        elif parsed["scale"] == "percent":
            checks["scale_consistent"] = bool(PERCENT_WORDS.search(question["question"] + " " + ev_text))
        else:
            checks["scale_consistent"] = None
    elif parsed["answer"]:
        doc_norm = _norm(ev_text)
        checks["answer_grounded"] = all(_norm(a) in doc_norm for a in parsed["answer"] if _norm(a))
    elif parsed["items"] is not None:
        checks["items_nonempty"] = len(parsed["items"]) > 0

    applicable = [v for v in checks.values() if v is not None]
    score = sum(applicable) / len(applicable) if applicable else 0.0
    return checks, score
