"""How well does example RAG match each dev question? (CPU only, ~1 min.)

For the k retrieved training examples of every dev question, the share that have the same answer
type, the same scale and (arithmetic only) the same kind of calculation as the dev question's gold
answer. Used to choose the template weight on dev; the test set is not touched.

    python scripts/retrieval_check.py
"""
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from finsight.data import load_split
from finsight.retrieve import ExampleRetriever

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def operation(derivation):
    """Rough kind of calculation from a gold derivation string."""
    d = derivation.replace(" ", "")
    if not d:
        return "none"
    if re.search(r"\)/[\d.,]+", d) and "-" in d and d.count("/") == 1:
        return "pct_change"
    if re.search(r"/(2|3|4)$", d):
        return "average"
    if "/" in d:
        return "ratio"
    if "-" in d:
        return "difference"
    if "+" in d:
        return "sum"
    return "other"


def evaluate(r, dev_docs, dev_q, k, use_doc):
    t = s = o = n = no = 0
    for q in dev_q:
        ex = [e for e, _ in r.nearest(q, k, doc=dev_docs[q["doc_uid"]] if use_doc else None)]
        t += sum(e["answer_type"] == q["answer_type"] for e in ex)
        s += sum(e["scale"] == q["scale"] for e in ex)
        n += len(ex)
        if q["answer_type"] == "arithmetic":
            op = operation(q["derivation"])
            o += sum(operation(e["derivation"]) == op for e in ex)
            no += len(ex)
    return 100 * t / n, 100 * s / n, 100 * o / no


def main(k=4):
    train_docs, train_q = load_split(os.path.join(DATA, "tatdqa_train.json.gz"))
    dev_docs, dev_q = load_split(os.path.join(DATA, "tatdqa_dev.json.gz"))
    r = ExampleRetriever(train_docs, train_q)
    print(f"{'setting':>28} | same type | same scale | same calculation (arith.)")
    for w, ub in ((0.0, 0.0), (1.0, 0.0), (1.0, 0.05), (1.0, 0.1), (1.0, 0.2), (1.0, 0.3)):
        r.w, r.unit_bonus = w, ub
        t, s, o = evaluate(r, dev_docs, dev_q, k, use_doc=ub > 0)
        print(f"{f'template {w}, unit bonus {ub}':>28} | {t:8.1f}% | {s:9.1f}% | {o:6.1f}%")


if __name__ == "__main__":
    main()
