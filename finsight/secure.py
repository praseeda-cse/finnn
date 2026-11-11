"""Secure question answering over PDFs: access control, integrity scan, gate and a tamper-evident log.

    ingest:  hash the file → integrity scan → parse → remove hidden words
    answer:  check access → solve.answer → gate → release only a Verified answer → log every step

The audit log is append-only JSON lines, each entry holding the hash of the one before it, so an
edited or deleted entry breaks the chain (verify_log finds where). It cannot stop someone rewriting
the whole file; for that the last hash would have to be stored elsewhere.
"""
import hashlib
import json
import os
import time

from . import gate, integrity, solve
from .prompt import parse_output

GENESIS = "0" * 64


def _entry_hash(entry):
    body = {k: v for k, v in entry.items() if k != "hash"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def verify_log(path):
    """(True, n_entries) if the chain is intact, else (False, index of the first bad entry)."""
    previous, count = GENESIS, 0
    if not os.path.exists(path):
        return True, 0
    with open(path, encoding="utf-8") as f:
        for count, line in enumerate(f):
            entry = json.loads(line)
            if entry["prev"] != previous or entry["hash"] != _entry_hash(entry):
                return False, count
            previous = entry["hash"]
        else:
            count = count + 1 if previous != GENESIS else 0
    return True, count


class SecureStore:
    def __init__(self, log_path, reader=None):
        self.log_path, self.reader, self.docs = log_path, reader, {}
        self.last = GENESIS
        if os.path.exists(log_path):
            with open(log_path, encoding="utf-8") as f:
                for line in f:
                    self.last = json.loads(line)["hash"]

    def _log(self, user, action, doc_id, **detail):
        entry = {"time": time.time(), "user": user, "action": action, "doc": doc_id, "detail": detail, "prev": self.last}
        entry["hash"] = self.last = _entry_hash(entry)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def ingest(self, path, owner, allowed=(), doc=None):
        """Scan and store a PDF for `owner` and the `allowed` users. Returns the document id (its hash).
        doc: an already parsed document; by default the PDF's own text layer is parsed."""
        report = integrity.scan(path, self.reader or integrity.OcrReader())
        if doc is None:
            from .demo import doc_from_pdf           # demo imports torch, which only the demo needs
            doc = doc_from_pdf(path)[0]
        doc_id = report["sha256"]
        self.docs[doc_id] = {"doc": integrity.drop_hidden(doc, report), "report": report, "users": {owner, *allowed}}
        self._log(owner, "ingest", doc_id, numbers=report["numbers"], suspects=len(report["suspects"]))
        return doc_id

    def answer(self, user, doc_id, question, client, **kwargs):
        """Answer for `user`. Only a Verified answer is released; `candidate` shows what was withheld."""
        stored = self.docs.get(doc_id)
        if stored is None or user not in stored["users"]:
            self._log(user, "denied", doc_id, question=question)      # same reply whether or not the document exists
            return {"verdict": "Denied", "answer": [], "reasons": ["no access to this document"]}
        result = solve.answer(client, stored["doc"], question, **kwargs)
        parsed = parse_output(solve.final_block(result["raw"]))
        checked = gate.check(stored["doc"], parsed, stored["report"])
        released = checked["verdict"] == "Verified"
        self._log(user, "answer", doc_id, question=question, verdict=checked["verdict"], reasons=checked["reasons"])
        return {**checked, "answer": result["answer"] if released else [], "scale": result["scale"] if released else "",
                "candidate": result["answer"], "evidence": result["evidence"], "agreement": result["agreement"]}
