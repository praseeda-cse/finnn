"""End-to-end checks of the security layer with a scripted model, so no GPU or server is needed.

    python -m pytest tests/test_security.py -q
"""
import json
import os
import sys

import fitz
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from finsight import gate, integrity, secure

SIZE = 11
ROWS = [("Total assets", "958,744", "937,510"), ("Total liabilities", "412,330", "398,115"), ("Revenue", "1,204,551", "1,120,884")]


def statement():
    """A small digital statement: one row per line, with a stated unit."""
    pdf = fitz.open()
    page = pdf.new_page(width=595, height=842)
    page.insert_text((72, 80), "Balance sheet (in thousands)", fontsize=SIZE)
    for i, (label, a, b) in enumerate(ROWS):
        y = 120 + 24 * i
        page.insert_text((72, y), label, fontsize=SIZE)
        page.insert_text((300, y), a, fontsize=SIZE)
        page.insert_text((400, y), b, fontsize=SIZE)
    return pdf


def write(pdf, tmp_path, name):
    path = str(tmp_path / name)
    pdf.save(path)
    return path


def box_of(page, text):
    return next(w[:4] for w in page.get_text("words") if w[4] == text)


def split_view(pdf, old, new):
    """Text layer says `new`, the page still shows `old`."""
    page = pdf[0]
    out = fitz.open()
    fresh = out.new_page(width=page.rect.width, height=page.rect.height)
    fresh.insert_image(fresh.rect, pixmap=page.get_pixmap(matrix=fitz.Matrix(3, 3)))
    for x0, y0, x1, y1, word, *_ in page.get_text("words"):
        fresh.insert_text((x0, y1 - 0.2 * (y1 - y0)), new if word == old else word, fontsize=(y1 - y0) * 0.8, render_mode=3)
    return out


def simple_doc():
    return {"pages": [{"size": [1240, 1754], "blocks": [
        {"id": 0, "words": ["Balance", "sheet", "(in", "thousands)"], "wboxes": [[10, 10, 50, 22], [55, 10, 90, 22], [95, 10, 110, 22], [115, 10, 180, 22]]},
        {"id": 1, "words": ["APAC", "958,744", "937,510"], "wboxes": [[10, 40, 50, 52], [300, 40, 360, 52], [400, 40, 460, 52]]},
    ]}]}


def parsed(program=None, answer=None, evidence=(1), scale="thousand"):
    return {"program": program, "answer": answer, "items": None, "evidence": [evidence] if isinstance(evidence, int) else list(evidence), "scale": scale}


class ScriptedClient:
    def __init__(self, reply):
        self.reply = reply

    def chat(self, *args, **kwargs):
        return self.reply


@pytest.fixture(scope="module")
def reader():
    return integrity.OcrReader()


# ---- gate rules (no OCR) ----
def test_clean_program_is_verified():
    assert gate.check(simple_doc(), parsed("958744 - 937510"))["verdict"] == "Verified"


def test_number_not_on_page_abstains():
    result = gate.check(simple_doc(), parsed("958744 - 111111"))
    assert result["verdict"] == "Abstain" and "111111" in result["reasons"][0]


def test_partial_number_does_not_bind():
    assert gate.check(simple_doc(), parsed("95874 - 937510"))["verdict"] == "Abstain"


def test_flagged_box_is_a_conflict():
    report = {"suspects": [{"page": 0, "box": [300, 40, 360, 52], "text": "958,744", "read": "958,144", "kind": "mismatch"}]}
    assert gate.check(simple_doc(), parsed("958744 - 937510"), report)["verdict"] == "Conflict"


def test_flag_elsewhere_does_not_block():
    report = {"suspects": [{"page": 0, "box": [10, 10, 50, 22], "text": "x", "read": "", "kind": "hidden"}]}
    assert gate.check(simple_doc(), parsed("958744 - 937510"), report)["verdict"] == "Verified"


def test_unit_contradicting_page_is_a_conflict():
    assert gate.check(simple_doc(), parsed("958744 - 937510", scale="million"))["verdict"] == "Conflict"


def test_times_100_needs_percent():
    assert gate.check(simple_doc(), parsed("(958744 - 937510) / 937510 * 100", scale=""))["verdict"] == "Conflict"
    assert gate.check(simple_doc(), parsed("(958744 - 937510) / 937510 * 100", scale="percent"))["verdict"] == "Verified"


def test_span_not_on_page_abstains():
    assert gate.check(simple_doc(), parsed(answer=["not in this file"], scale=""))["verdict"] == "Abstain"


def test_nothing_to_check_abstains():
    assert gate.check(simple_doc(), parsed())["verdict"] == "Abstain"


# ---- integrity scan (OCR) ----
def test_clean_statement_has_no_suspects(reader):
    report = integrity.scan(statement().tobytes(), reader)
    assert report["numbers"] >= 6 and report["suspects"] == []


def test_split_view_is_detected_at_the_right_word(reader):
    report = integrity.scan(split_view(statement(), "958,744", "958,144").tobytes(), reader)
    assert [s["text"] for s in report["suspects"]] == ["958,144"]
    assert report["suspects"][0]["kind"] == "mismatch" and integrity.digits(report["suspects"][0]["read"]) == "958744"


def test_invisible_and_white_text_are_hidden(reader):
    for kwargs in ({"render_mode": 3}, {"color": (1, 1, 1)}):
        pdf = statement()
        pdf[0].insert_text((72, 300), "marker", fontsize=SIZE, **kwargs)
        report = integrity.scan(pdf.tobytes(), reader)
        assert [(s["text"], s["kind"]) for s in report["suspects"]] == [("marker", "hidden")]


def test_drop_hidden_removes_words_before_the_model():
    doc = simple_doc()
    report = {"suspects": [{"page": 0, "box": [10, 40, 50, 52], "text": "APAC", "read": "", "kind": "hidden"}]}
    assert integrity.drop_hidden(doc, report)["pages"][0]["blocks"][1]["words"] == ["958,744", "937,510"]


def test_consistent_edit_passes_scan_known_limit(reader):
    pdf = statement()
    page = pdf[0]
    rect = fitz.Rect(box_of(page, "958,744"))
    page.add_redact_annot(rect, text="958,144", fontsize=SIZE - 1)
    page.apply_redactions()
    assert integrity.scan(pdf.tobytes(), reader)["suspects"] == []


# ---- whole flow ----
REPLY = "evidence: {row}\ntype: arithmetic\nprogram: {a} - 937510\nscale: thousand"


def rows_of(store, doc_id, number):
    for block in store.docs[doc_id]["doc"]["pages"][0]["blocks"]:
        if number in block["words"]:
            return block["id"]
    raise AssertionError(number)


def test_end_to_end(tmp_path, reader):
    log = str(tmp_path / "audit.jsonl")
    store = secure.SecureStore(log, reader)

    clean = store.ingest(write(statement(), tmp_path, "clean.pdf"), "asha", allowed=["ravi"])
    reply = REPLY.format(row=rows_of(store, clean, "958,744"), a="958,744")
    good = store.answer("ravi", clean, "Change in total assets?", ScriptedClient(reply))
    assert good["verdict"] == "Verified" and float(good["answer"][0]) == 21234 and good["scale"] == "thousand"

    denied = store.answer("mallory", clean, "Change in total assets?", ScriptedClient(reply))
    assert denied["verdict"] == "Denied" and denied["answer"] == []
    assert store.answer("mallory", "no-such-doc", "q", ScriptedClient(reply))["reasons"] == denied["reasons"]

    tampered = store.ingest(write(split_view(statement(), "958,744", "958,144"), tmp_path, "tampered.pdf"), "asha")
    reply = REPLY.format(row=rows_of(store, tampered, "958,144"), a="958,144")
    bad = store.answer("asha", tampered, "Change in total assets?", ScriptedClient(reply))
    assert bad["verdict"] == "Conflict" and bad["answer"] == [] and bad["candidate"]

    assert secure.verify_log(log) == (True, 6)


def test_log_edit_is_found(tmp_path):
    log = str(tmp_path / "audit.jsonl")
    store = secure.SecureStore(log)
    for i in range(3):
        store._log("asha", "event", "doc", i=i)
    lines = open(log, encoding="utf-8").read().splitlines()
    entry = json.loads(lines[1])
    entry["user"] = "ravi"
    lines[1] = json.dumps(entry)
    open(log, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    assert secure.verify_log(log) == (False, 1)
