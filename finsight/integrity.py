"""Numeric integrity scan: does the PDF's text layer say what the rendered page shows?

The text layer is what a text reader sees and the rendered page is what a person sees, and a tampered
PDF can make them differ. Every numeric word is cropped from the render and read back with OCR, and
only exact digit equality passes. Words that are not visibly drawn at all are reported as hidden.
Box coordinates are in the ~1240px frame of the dataset.

An edit that changes both layers consistently passes this scan by design; see scripts/tamper_eval.py.
"""
import hashlib
import re

import fitz
import numpy as np

TARGET_WIDTH = 1240       # same coordinate frame as demo.doc_from_pdf
NUMERIC = re.compile(r"^[(\[$€£¥+\-–−]*\d[\d,.]*[)\]%]*[.,;:]*$")
ZOOMS = (3.0, 6.0)        # second pass only for words the first pass disagrees with
PAD = 2.0                 # PDF points around a word when cropping
INVISIBLE_RANGE = 30      # grey-level range below which a word's box shows no ink
MIN_WORD_HEIGHT = 2.0     # PDF points; anything shorter cannot be read by a person


def digits(text):
    return re.sub(r"\D", "", text)


class OcrReader:
    """Recognition only: the crop already is the text line."""

    def __init__(self):
        from rapidocr import RapidOCR
        self.engine = RapidOCR(params={"Global.log_level": "error"})

    def __call__(self, image):
        result = self.engine(image, use_det=False, use_cls=False, use_rec=True)
        return "".join(result.txts) if result.txts else ""


def _image(pix):
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)[:, :, :3]


def _crop(page, rect, zoom):
    return _image(page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=rect + (-PAD, -PAD, PAD, PAD)))


def _hidden_spans(page):
    """Boxes of invisible-mode text, on pages where most text is drawn normally. On a scanned page
    with an OCR layer all text is invisible by construction, so nothing is reported."""
    spans = page.get_texttrace()
    invisible = [fitz.Rect(s["bbox"]) for s in spans if s["type"] == 3 or s["opacity"] == 0]
    return invisible if spans and len(invisible) < 0.5 * len(spans) else []


def _visible(render, rect, zoom):
    x0, y0, x1, y1 = (int(max(v, 0) * zoom) for v in rect)
    patch = render[y0:y1, x0:x1].mean(axis=2)
    return patch.size > 0 and patch.max() - patch.min() >= INVISIBLE_RANGE


def scan_page(page, reader):
    """Suspect words of one page, in PDF points: [(rect, text, read, kind), ...], plus the numeric count."""
    zoom = ZOOMS[0]
    render = _image(page.get_pixmap(matrix=fitz.Matrix(zoom, zoom)))
    hidden = _hidden_spans(page)
    suspects, numbers = [], 0
    for x0, y0, x1, y1, word, *_ in page.get_text("words"):
        if not word.strip():      # spacing characters draw no ink
            continue
        rect = fitz.Rect(x0, y0, x1, y1)
        if y1 - y0 < MIN_WORD_HEIGHT or not _visible(render, rect, zoom) \
                or any(rect.intersects(h) and (rect & h).get_area() > 0.5 * rect.get_area() for h in hidden):
            suspects.append((rect, word, "", "hidden"))
            continue
        if not NUMERIC.match(word):
            continue
        numbers += 1
        want = digits(word)
        reads = []
        for z in ZOOMS:
            reads.append(digits(reader(_crop(page, rect, z))))
            if reads[-1] == want:
                break
        if reads[-1] != want:
            suspects.append((rect, word, reads[-1], "mismatch" if any(reads) else "unreadable"))
    return suspects, numbers


def scan(source, reader=None, max_pages=3):
    """Scan a PDF (path or bytes). Returns {sha256, pages, numbers, suspects}, boxes in the dataset frame."""
    reader = reader or OcrReader()
    data = source if isinstance(source, (bytes, bytearray)) else open(source, "rb").read()
    pdf = fitz.open(stream=data, filetype="pdf")
    out, numbers = [], 0
    for index, page in enumerate(list(pdf)[:max_pages]):
        scale = TARGET_WIDTH / page.rect.width
        found, count = scan_page(page, reader)
        numbers += count
        out += [{"page": index, "box": [round(v * scale, 1) for v in rect], "text": text, "read": read, "kind": kind}
                for rect, text, read, kind in found]
    return {"sha256": hashlib.sha256(data).hexdigest(), "pages": min(len(pdf), max_pages), "numbers": numbers, "suspects": out}


def overlaps(box, other, threshold=0.3):
    """True when the shared area covers at least `threshold` of the smaller box."""
    width = min(box[2], other[2]) - max(box[0], other[0])
    height = min(box[3], other[3]) - max(box[1], other[1])
    if width <= 0 or height <= 0:
        return False
    smaller = min((box[2] - box[0]) * (box[3] - box[1]), (other[2] - other[0]) * (other[3] - other[1]))
    return width * height >= threshold * max(smaller, 1e-6)


def drop_hidden(doc, report):
    """Remove hidden words from a parsed document, so they never reach the model."""
    hidden = [s for s in report["suspects"] if s["kind"] == "hidden"]
    for index, page in enumerate(doc["pages"]):
        marks = [s["box"] for s in hidden if s["page"] == index]
        for block in page["blocks"]:
            keep = [i for i, box in enumerate(block["wboxes"]) if not any(overlaps(box, m) for m in marks)]
            block["words"] = [block["words"][i] for i in keep]
            block["wboxes"] = [block["wboxes"][i] for i in keep]
        page["blocks"] = [b for b in page["blocks"] if b["words"]]
    return doc
