"""Defensive evaluation of the integrity scan and the gate, on copies we alter ourselves. No model, no GPU.

1. Benign cost: gold answers through the gate on every dev question (how many correct answers it blocks).
2. False alarms: the scan on every clean dev PDF.
3. Detection: for a seeded sample of dev pages, alter one gold operand in each way below.

    split_view        text layer changed, page looks untouched
    visible_overlay   page changed, text layer untouched
    consistent_edit   both changed the same way (expected to pass: the scan's known limit)
    hidden_invisible  extra text in invisible mode     hidden_white  extra white text on white
    twin              split_view with no change at all (control for false alarms from the conversion)

    python scripts/tamper_eval.py --docs ../fp/dataset/tatdqa_docs_dev/dev --n 100

Dev only: the test split stays for the single final run. Results go to runs/tamper_eval.json.
"""
import argparse
import ast
import copy
import json
import math
import os
import random
import re
import sys
import time

import fitz

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from finsight.data import load_split
from finsight.gate import check
from finsight.integrity import NUMERIC, TARGET_WIDTH, OcrReader, digits, drop_hidden, overlaps, scan
from finsight.prompt import gold_target, parse_output

CONFUSABLE = {"0": "8", "8": "3", "3": "8", "1": "7", "7": "1", "5": "6", "6": "5", "2": "7", "4": "9", "9": "4"}
MARKER = "INTEGRITY TEST MARKER 12345"


def wilson(hits, n, z=1.96):
    if not n:
        return [0.0, 0.0]
    p, d = hits / n, 1 + z * z / n
    centre, half = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(centre - half, 4), round(centre + half, 4)]


def alter(word, rng, confusable):
    """Change one digit, keeping the layout of the number."""
    places = [i for i, c in enumerate(word) if c.isdigit()]
    i = rng.choice(places)
    new = CONFUSABLE[word[i]] if confusable else rng.choice([d for d in "0123456789" if d != word[i]])
    return word[:i] + new + word[i + 1:]


def find_target(page, facts):
    """A numeric word on the page equal to a gold operand: (x0, y0, x1, y1, word) or None."""
    wanted = {digits(f) for f in facts if len(digits(f)) >= 2}
    for x0, y0, x1, y1, word, *_ in page.get_text("words"):
        if NUMERIC.match(word) and digits(word) in wanted:
            return x0, y0, x1, y1, word
    return None


def _single_page(data, index):
    pdf = fitz.open(stream=data, filetype="pdf")
    pdf.select([index])
    return pdf


def split_view(data, index, target, new):
    """Rendered page as the background, with an invisible text layer on top."""
    src = _single_page(data, index)
    page = src[0]
    out = fitz.open()
    fresh = out.new_page(width=page.rect.width, height=page.rect.height)
    fresh.insert_image(fresh.rect, pixmap=page.get_pixmap(matrix=fitz.Matrix(2, 2)))
    for x0, y0, x1, y1, word, *_ in page.get_text("words"):
        text = new if target is not None and (x0, y0, x1, y1) == target[:4] else word
        fresh.insert_text((x0, y1 - 0.2 * (y1 - y0)), text, fontsize=max(y1 - y0, 1) * 0.8, render_mode=3)
    return out.tobytes()


def visible_overlay(data, index, target, new):
    src = _single_page(data, index)
    page, rect = src[0], fitz.Rect(target[:4])
    patch = fitz.open()
    p = patch.new_page(width=rect.width + 4, height=rect.height + 2)
    p.insert_text((1, rect.height * 0.8), new, fontsize=rect.height * 0.8)
    page.insert_image(rect + (-1, 0, 3, 2), pixmap=p.get_pixmap(matrix=fitz.Matrix(4, 4)), overlay=True)
    return src.tobytes()


def consistent_edit(data, index, target, new):
    src = _single_page(data, index)
    page, rect = src[0], fitz.Rect(target[:4])
    page.add_redact_annot(rect, text=new, fontsize=rect.height * 0.75, align=fitz.TEXT_ALIGN_RIGHT)
    page.apply_redactions()
    return src.tobytes()


def hidden(data, index, mode):
    src = _single_page(data, index)
    page = src[0]
    kwargs = {"render_mode": 3} if mode == "invisible" else {"color": (1, 1, 1)}
    page.insert_text((36, 20), MARKER, fontsize=8, **kwargs)
    return src.tobytes()


def benign_gate(docs, questions, reports=None):
    """Gold answers through the gate on untampered files. With reports, each document is first
    cleaned of hidden words and checked against its real scan, as in secure.SecureStore."""
    verdicts, reasons = {}, {}
    cleaned = {u: drop_hidden(copy.deepcopy(docs[u]), r) for u, r in (reports or {}).items()}
    for q in questions:
        target = gold_target(q)
        if target is None:
            continue
        uid = q["doc_uid"]
        result = check(cleaned.get(uid, docs[uid]), parse_output(target), (reports or {}).get(uid))
        verdicts[result["verdict"]] = verdicts.get(result["verdict"], 0) + 1
        for r in result["reasons"]:
            key = re.sub(r"[\d.,]+", "N", r)
            reasons[key] = reasons.get(key, 0) + 1
    total = sum(verdicts.values())
    return {"n": total, "verdicts": verdicts, "verified_rate": round(verdicts.get("Verified", 0) / total, 4),
            "top_reasons": sorted(reasons.items(), key=lambda x: -x[1])[:6]}


def attacked_hit(report, target_box, index_scale):
    box = [v * index_scale for v in target_box]
    return any(overlaps(box, s["box"]) for s in report["suspects"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", required=True, help="folder with <uid>.pdf for the dev split")
    ap.add_argument("--n", type=int, default=100, help="pages to attack")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="only the first N clean PDFs (quick check)")
    ap.add_argument("--clean-only", action="store_true", help="stop after the clean pages; writes runs/tamper_clean.json")
    args = ap.parse_args()

    root = os.path.join(os.path.dirname(__file__), "..")
    docs, questions = load_split(os.path.join(root, "data", "tatdqa_dev.json.gz"))
    results = {"benign_gate": benign_gate(docs, questions)}
    print(results["benign_gate"], flush=True)

    reader = OcrReader()
    uids = sorted(u for u in docs if os.path.exists(os.path.join(args.docs, f"{u}.pdf")))
    if args.limit:
        uids = uids[:args.limit]
    start, pages, numbers, flagged_pages, kinds, examples, reports = time.time(), 0, 0, 0, {}, [], {}
    for uid in uids:
        report = reports[uid] = scan(os.path.join(args.docs, f"{uid}.pdf"), reader)
        pages += report["pages"]
        numbers += report["numbers"]
        flagged_pages += len({s["page"] for s in report["suspects"]})
        for s in report["suspects"]:
            kinds[s["kind"]] = kinds.get(s["kind"], 0) + 1
            if len(examples) < 25:
                examples.append([s["text"], s["read"], s["kind"]])
    mismatches = kinds.get("mismatch", 0) + kinds.get("unreadable", 0)
    results["clean"] = {"docs": len(uids), "pages": pages, "numbers": numbers, "flagged_pages": flagged_pages,
                        "page_fp_rate": round(flagged_pages / max(pages, 1), 4),
                        "page_fp_ci": wilson(flagged_pages, pages), "suspects_by_kind": kinds,
                        "token_fp_rate": round(mismatches / max(numbers, 1), 5), "examples": examples,
                        "seconds_per_page": round((time.time() - start) / max(pages, 1), 2)}
    print(results["clean"], flush=True)
    results["benign_gate_scanned"] = benign_gate(docs, [q for q in questions if q["doc_uid"] in reports], reports)
    print(results["benign_gate_scanned"], flush=True)
    if args.clean_only:
        os.makedirs(os.path.join(root, "runs"), exist_ok=True)
        with open(os.path.join(root, "runs", "tamper_clean.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=1, ensure_ascii=False)
        return

    rng = random.Random(args.seed)
    by_doc = {}
    for q in questions:
        if q["answer_type"] == "arithmetic" and q["doc_uid"] in uids:
            by_doc.setdefault(q["doc_uid"], []).append(q)
    order = sorted(by_doc)
    rng.shuffle(order)
    tally, tried = {}, 0

    def record(name, detected, any_flag):
        t = tally.setdefault(name, {"n": 0, "detected": 0, "page_flagged": 0})
        t["n"] += 1
        t["detected"] += int(detected)
        t["page_flagged"] += int(any_flag)

    for uid in order:
        if tried >= args.n:
            break
        data = open(os.path.join(args.docs, f"{uid}.pdf"), "rb").read()
        pdf = fitz.open(stream=data, filetype="pdf")
        chosen = None
        for q in by_doc[uid]:
            facts = ast.literal_eval(q["facts"]) if isinstance(q["facts"], str) else q["facts"]
            for index, page in enumerate(list(pdf)[:3]):
                found = find_target(page, facts)
                if found:
                    chosen = (index, found)
                    break
            if chosen:
                break
        if not chosen:
            continue
        tried += 1
        index, target = chosen
        scale = TARGET_WIDTH / pdf[index].rect.width
        for strategy in ("random", "confusable"):
            new = alter(target[4], rng, strategy == "confusable")
            for name, build in (("split_view", split_view), ("visible_overlay", visible_overlay), ("consistent_edit", consistent_edit)):
                report = scan(build(data, index, target, new), reader, max_pages=1)
                record(f"{name}/{strategy}", attacked_hit(report, target[:4], scale), bool(report["suspects"]))
        report = scan(split_view(data, index, None, None), reader, max_pages=1)
        record("twin/control", False, bool(report["suspects"]))
        for mode in ("invisible", "white"):
            report = scan(hidden(data, index, mode), reader, max_pages=1)
            record(f"hidden_{mode}", any(s["kind"] == "hidden" and s["text"] in MARKER.split() for s in report["suspects"]), bool(report["suspects"]))
        if tried % 10 == 0:
            print(f"{tried}/{args.n} pages attacked", flush=True)

    for t in tally.values():
        t["recall"] = round(t["detected"] / t["n"], 4)
        t["recall_ci"] = wilson(t["detected"], t["n"])
        t["page_flag_rate"] = round(t["page_flagged"] / t["n"], 4)
    results["attacks"] = tally
    os.makedirs(os.path.join(root, "runs"), exist_ok=True)
    with open(os.path.join(root, "runs", "tamper_eval.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1, ensure_ascii=False)
    for name, t in sorted(tally.items()):
        print(f"{name:28s} n={t['n']:4d}  detected {t['recall']:.1%} {t['recall_ci']}  any flag {t['page_flag_rate']:.1%}")


if __name__ == "__main__":
    main()
