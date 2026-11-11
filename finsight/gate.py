"""Provenance gate: an answer is released only if every number in it can be traced and trusted.

solve.grounded asks whether a number occurs anywhere in the document. The gate asks more, for each
number: which box is it, is that box in a row the model cited, did the integrity scan flag that box,
and does the unit agree with the page? Rules only, no model. Verdicts:

    Verified  every number is bound to a clean box and the unit is consistent
    Conflict  a number sits on a flagged box, or the unit contradicts the page or the program
    Abstain   a number or span cannot be found on the page, or there is nothing to check
"""
import re

from .integrity import overlaps
from .solve import NUM, SMALL_CONSTANTS

THOUSAND = re.compile(r"in\s+thousands|\(000s?\)|\$\s*'?000|'000", re.I)
MILLION = re.compile(r"in\s+millions|\$\s*m\b|\$\s*million", re.I)
TIMES_100 = re.compile(r"\*\s*100(?:\.0*)?\s*\)*\s*$")


def _numbers(text):
    return {n.replace(",", "").rstrip(".") for n in NUM.findall(text)}


def _blocks(doc):
    for index, page in enumerate(doc["pages"]):
        for block in page["blocks"]:
            yield index, block


def _hits(doc, number):
    return [{"page": index, "block": block["id"], "box": box}
            for index, block in _blocks(doc)
            for word, box in zip(block["words"], block["wboxes"]) if number in _numbers(word)]


def _flagged(hit, report):
    return report is not None and any(s["page"] == hit["page"] and overlaps(hit["box"], s["box"]) for s in report["suspects"])


def _declared_unit(doc):
    text = " ".join(w for _, block in _blocks(doc) for w in block["words"])
    found = {unit for unit, pattern in (("thousand", THOUSAND), ("million", MILLION)) if pattern.search(text)}
    return found.pop() if len(found) == 1 else None


def scale_problem(doc, parsed):
    """Reason the predicted unit is inconsistent, or None. Only clear contradictions are reported."""
    scale, program = parsed.get("scale") or "", parsed.get("program") or ""
    if program and TIMES_100.search(program) and scale != "percent":
        return f"program multiplies by 100 but the unit is '{scale or 'none'}'"
    declared = _declared_unit(doc)
    if program and declared and scale in ("thousand", "million") and scale != declared:
        return f"page reports in {declared}s but the unit is '{scale}'"
    return None


def check(doc, parsed, report=None):
    """Gate one parsed model output. Returns {verdict, operands, reasons}."""
    cited = set(parsed.get("evidence") or [])
    if parsed.get("program"):
        numbers = _numbers(parsed["program"]) - SMALL_CONSTANTS
    else:
        numbers = set().union(*[_numbers(a) for a in parsed.get("answer") or []], *[_numbers(i) for i in parsed.get("items") or []])
    if not parsed.get("program") and not parsed.get("answer") and not parsed.get("items"):
        return {"verdict": "Abstain", "operands": [], "reasons": ["no answer to check"]}

    operands, reasons, conflict, missing = [], [], False, False
    for number in sorted(numbers):
        hits = _hits(doc, number)
        in_cited = [h for h in hits if h["block"] in cited]
        chosen = in_cited or hits
        if not chosen:
            operands.append({"value": number, "status": "not on page"})
            reasons.append(f"{number} is not on the page")
            missing = True
            continue
        # A cited row is what the answer rests on, so one flagged copy there is enough.
        bad = any(_flagged(h, report) for h in chosen) if in_cited else all(_flagged(h, report) for h in chosen)
        clean = next((h for h in chosen if not _flagged(h, report)), chosen[0])
        operands.append({"value": number, "page": clean["page"], "block": clean["block"], "box": clean["box"],
                         "cited": bool(in_cited), "status": "flagged" if bad else "bound"})
        if bad:
            reasons.append(f"{number} sits on a box that failed the integrity scan")
            conflict = True

    text = "".join(w for _, block in _blocks(doc) for w in block["words"]).lower()
    for span in parsed.get("answer") or []:
        if "".join(span.lower().split()) not in text:
            reasons.append("an answer span is not on the page")
            missing = True

    problem = scale_problem(doc, parsed)
    if problem:
        reasons.append(problem)
        conflict = True
    verdict = "Conflict" if conflict else "Abstain" if missing else "Verified"
    return {"verdict": verdict, "operands": operands, "reasons": reasons}
