"""Automatic risk report: when a statement is loaded, FinSight asks a fixed set of risk questions
without the user typing anything ("Can the company pay its short-term debts?", "Is it relying on
debt?", ...) and answers each with a value, the formula, a Low / Medium / High flag and the rows used.

How it stays trustworthy:
1. One model call *reads* the needed line items (revenue, current assets, ...) for each year, copying
   numbers as printed and naming the row they came from. The model never calculates.
2. Every value is checked against the cited row; a number that is not on that row is dropped.
3. Python computes every ratio and applies the thresholds. Items the document does not contain are
   reported as "not in this document" instead of being guessed.
4. Separately, a table scan with no model flags line items that changed by 50% or more year on year.

Thresholds are common rules of thumb from financial-statement analysis, not industry-specific
limits; the report says so.
"""
import json
import re

from .data import block_text, render_doc
from .tatqa_utils import to_number

ITEMS = {
    "revenue": "total revenue / net sales / turnover",
    "operating_income": "operating income / operating profit / EBIT",
    "net_income": "net income / net profit or loss for the year",
    "interest_expense": "interest expense / finance costs",
    "current_assets": "total current assets",
    "current_liabilities": "total current liabilities",
    "total_assets": "total assets",
    "total_liabilities": "total liabilities",
    "total_equity": "total shareholders' (stockholders') equity",
    "total_debt": "total borrowings / total debt",
    "operating_cash_flow": "net cash provided by (used in) operating activities",
}

EXTRACT_SYSTEM = """You read a financial document given as numbered rows ([id] text); table cells are separated by " | ".
Find the line items below for every year the document shows. Copy each number exactly as printed, keeping parentheses for negative values. Never calculate, never estimate. Leave an item out if the document does not state it.
Reply with JSON only, in this form:
{"unit": "<thousand | million | billion | empty>", "items": {"<item>": {"<year>": {"value": "<number as printed>", "row": <row id>}}}}
Items:
""" + "\n".join(f"- {k}: {v}" for k, v in ITEMS.items())

# TAT-DQA questions about the quantities the risk report reads (218 of 1,663 test questions): the
# pipeline's accuracy on them is the automatic evaluation of the risk report's inputs.
RISK_QUESTION = re.compile(
    r"current (assets|liabilities)|current ratio|\bdebt\b|borrowing|liabilit|equity|interest expense|interest coverage|"
    r"\bmargin\b|net (income|loss|profit)|operating (income|profit|loss)|cash (flow|provided by|used in) operating|"
    r"operating activities", re.I)


def risk_questions(questions):
    return [q for q in questions if RISK_QUESTION.search(q["question"])]


YEAR_CELL = re.compile(r"^(?:fiscal\s+|fy\s*)?((?:19|20)\d{2})$", re.I)
NUM_CELL = re.compile(r"\(?-?\$?\s*\d[\d,]*(?:\.\d+)?\s*\)?")


def _number(text):
    """'(1,234)' -> -1234.0, '$ 5,678' -> 5678.0; None if not a number."""
    t = str(text).strip()
    neg = t.startswith("(") and t.endswith(")") or t.startswith("-")
    v = to_number(re.sub(r"[()$\s-]", "", t))
    return None if v is None else (-abs(v) if neg else v)


def _parse_json(text):
    text = text[text.find("{"): text.rfind("}") + 1]
    text = re.sub(r",\s*([}\]])", r"\1", text)   # trailing commas
    try:
        return json.loads(text)
    except Exception:
        return {}


def extract(client, doc, max_tokens=1500):
    """Model reads the line items; every value is checked against the row it cites.
    Returns ({item: {year: (value, row id)}}, unit)."""
    msgs = [{"role": "system", "content": EXTRACT_SYSTEM},
            {"role": "user", "content": f"Document:\n{render_doc(doc)}"}]
    data = _parse_json(client.chat(msgs, 0.0, max_tokens))
    rows = {b["id"]: {n.replace(",", "").rstrip(".") for w in b["words"] for n in re.findall(r"\d[\d,]*(?:\.\d+)?", w)}
            for p in doc["pages"] for b in p["blocks"]}
    found = {}
    for item, years in (data.get("items") or {}).items():
        if item not in ITEMS or not isinstance(years, dict):
            continue
        for year, cell in years.items():
            try:
                raw, row = str(cell["value"]), int(cell["row"])
            except (KeyError, TypeError, ValueError):
                continue
            value = _number(raw)
            digits = re.sub(r"[^\d.]", "", raw).rstrip(".")
            if value is None or not digits or row not in rows or digits not in rows[row]:
                continue                                   # not on the cited row: dropped
            m = re.search(r"(?:19|20)\d{2}", str(year))
            if m:
                found.setdefault(item, {})[int(m.group())] = (value, row)
    unit = str(data.get("unit", "")).lower()
    return found, unit if unit in ("thousand", "million", "billion") else ""


def _fmt(v):
    return f"{v:,.2f}".rstrip("0").rstrip(".")


def _level(value, high, medium):
    """high/medium: functions value -> bool, checked in that order."""
    return "High" if high(value) else "Medium" if medium(value) else "Low"


def checks(found):
    """The automatic risk questions, answered from the verified values with Python."""
    years = sorted({y for v in found.values() for y in v}, reverse=True)
    if not years:
        return [], None, None
    y = years[0]
    p = years[1] if len(years) > 1 else None
    get = lambda item, year: found.get(item, {}).get(year, (None, None))
    out = []

    def add(question, metric, value, formula, level, why, rows):
        out.append({"question": question, "metric": metric, "value": round(value, 4), "formula": formula,
                    "level": level, "why": why, "evidence": sorted({r for r in rows if r is not None})})

    (ca, r1), (cl, r2) = get("current_assets", y), get("current_liabilities", y)
    if ca is not None and cl:
        v = ca / cl
        add("Can the company pay its short-term obligations?", f"Current ratio {y}", v, f"{_fmt(ca)} / {_fmt(cl)}",
            _level(v, lambda x: x < 1.0, lambda x: x < 1.5), "below 1.0 is high risk, 1.0-1.5 medium", [r1, r2])

    (eq, r2) = get("total_equity", y)
    debt, r1 = get("total_debt", y)
    name = "Debt-to-equity"
    if debt is None:
        (debt, r1), name = get("total_liabilities", y), "Liabilities-to-equity"
    if debt is not None and eq is not None and eq != 0:
        v = debt / eq
        level = "High" if eq < 0 else _level(v, lambda x: x > 2.0, lambda x: x > 1.0)
        add("Does the company rely heavily on borrowed money?", f"{name} {y}", v, f"{_fmt(debt)} / {_fmt(eq)}",
            level, "negative equity or above 2.0 is high risk, 1.0-2.0 medium", [r1, r2])

    (oi, r1), (ie, r2) = get("operating_income", y), get("interest_expense", y)
    if oi is not None and ie:
        v = oi / abs(ie)
        add("Can operating profit cover interest payments?", f"Interest coverage {y}", v, f"{_fmt(oi)} / {_fmt(abs(ie))}",
            _level(v, lambda x: x < 1.5, lambda x: x < 3.0), "below 1.5 is high risk, 1.5-3.0 medium", [r1, r2])

    (ni, r1), (rev, r2) = get("net_income", y), get("revenue", y)
    if ni is not None and rev:
        v = ni / rev * 100
        add("Is the company profitable?", f"Net profit margin {y} (%)", v, f"{_fmt(ni)} / {_fmt(rev)} * 100",
            _level(v, lambda x: x < 0, lambda x: x < 5), "a loss is high risk, a margin below 5% medium", [r1, r2])

    if p is not None:
        (rev_p, r3) = get("revenue", p)
        if rev is not None and rev_p:
            v = (rev - rev_p) / abs(rev_p) * 100
            add("Are sales falling?", f"Revenue change {p}-{y} (%)", v, f"({_fmt(rev)} - {_fmt(rev_p)}) / {_fmt(abs(rev_p))} * 100",
                _level(v, lambda x: x < -10, lambda x: x < 0), "a fall of more than 10% is high risk, any fall medium", [r2, r3])
        (ni_p, r4) = get("net_income", p)
        if ni is not None and rev and ni_p is not None and rev_p:
            v = ni / rev * 100 - ni_p / rev_p * 100
            add("Is profitability getting worse?", f"Net margin change {p}-{y} (percentage points)", v,
                f"{_fmt(ni)} / {_fmt(rev)} * 100 - {_fmt(ni_p)} / {_fmt(rev_p)} * 100",
                _level(v, lambda x: x < -5, lambda x: x < 0), "a drop of more than 5 points is high risk, any drop medium",
                [r1, r2, r3, r4])

    (ocf, r5) = get("operating_cash_flow", y)
    if ocf is not None and ni:
        v = ocf / ni
        level = "High" if ocf < 0 else ("Medium" if ni > 0 and v < 0.8 else "Low")
        add("Are profits backed by cash?", f"Operating cash flow / net income {y}", v, f"{_fmt(ocf)} / {_fmt(ni)}",
            level, "negative operating cash flow is high risk, below 0.8x profit medium", [r5, r1])
    return out, y, p


def unusual_changes(doc, threshold=50.0, top=5):
    """No model: rows of a year-column table whose value moved by threshold% or more year on year."""
    flags, years = [], None
    for page in doc["pages"]:
        for b in page["blocks"]:
            cells = [c.strip() for c in block_text(b).split(" | ")]
            ys = [int(m.group(1)) for c in cells if (m := YEAR_CELL.match(c))]
            if len(ys) >= 2 and len(ys) >= len(cells) - 1:
                years = ys                                # a header row: columns are these years
                continue
            if not years or "%" in "".join(cells):
                continue
            nums = [_number(c) for c in cells if NUM_CELL.fullmatch(c)]
            nums = [n for n in nums if n is not None]
            label = cells[0] if cells and _number(cells[0]) is None else ""
            if not label or len(nums) != len(years):
                continue
            cur, prev = (nums[years.index(max(years))], nums[years.index(sorted(set(years))[-2])]) if len(set(years)) >= 2 else (None, None)
            if cur is None or not prev:
                continue
            change = (cur - prev) / abs(prev) * 100
            if abs(change) >= threshold:
                flags.append({"item": label, "from": prev, "to": cur, "change_pct": round(change, 1),
                              "years": f"{sorted(set(years))[-2]}-{max(years)}", "row": b["id"]})
    return sorted(flags, key=lambda f: -abs(f["change_pct"]))[:top]


def report(client, doc):
    """Full automatic risk report for one document."""
    found, unit = extract(client, doc)
    results, year, prev = checks(found)
    asked = ["Can the company pay its short-term obligations?", "Does the company rely heavily on borrowed money?",
             "Can operating profit cover interest payments?", "Is the company profitable?", "Are sales falling?",
             "Is profitability getting worse?", "Are profits backed by cash?"]
    answered = {c["question"] for c in results}
    levels = [c["level"] for c in results]
    overall = ("High" if "High" in levels else "Medium" if "Medium" in levels else "Low") if levels else "Not enough data"
    return {"year": year, "previous_year": prev, "unit": unit, "overall": overall, "checks": results,
            "not_in_document": [q for q in asked if q not in answered],
            "unusual_changes": unusual_changes(doc), "values": found}


def render_html(rep):
    colour = {"High": "#c53030", "Medium": "#b7791f", "Low": "#2f855a", "Not enough data": "#4a5568"}
    rows = "".join(
        f"<tr><td>{c['question']}</td><td>{c['metric']}: <b>{_fmt(c['value'])}</b><br><small>{c['formula']}</small></td>"
        f"<td style='color:{colour[c['level']]}'><b>{c['level']}</b><br><small>{c['why']}</small></td>"
        f"<td>{', '.join(map(str, c['evidence']))}</td></tr>" for c in rep["checks"])
    unusual = "".join(f"<li>{u['item']}: {_fmt(u['from'])} → {_fmt(u['to'])} ({u['change_pct']:+}%, {u['years']}, row {u['row']})</li>"
                      for u in rep["unusual_changes"])
    missing = "".join(f"<li>{q}</li>" for q in rep["not_in_document"])
    unit = f" (values in {rep['unit']}s)" if rep["unit"] else ""
    return (f"<h3>Automatic risk report{unit}: overall <span style='color:{colour[rep['overall']]}'>{rep['overall']}</span></h3>"
            f"<table border=1 cellpadding=4><tr><th>Question</th><th>Answer</th><th>Risk</th><th>Rows</th></tr>{rows}</table>"
            + (f"<p><b>Large year-on-year changes</b></p><ul>{unusual}</ul>" if unusual else "")
            + (f"<p><b>Not in this document</b></p><ul>{missing}</ul>" if missing else "")
            + "<p><small>Thresholds are general rules of thumb, not industry-specific limits.</small></p>")
