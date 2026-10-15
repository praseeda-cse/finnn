"""Demo helpers. Read any digital PDF into the same block format as TAT-DQA, answer a question
with TAT-LLM + the Python executor, and draw the grounded evidence rows on the page."""
import html

import torch

from . import tatllm
from .data import block_text

TARGET_WIDTH = 1240  # TAT-DQA page coordinate width; column-gap threshold is tuned for it


def doc_from_pdf(path, max_pages=3):
    """One block per text line (a table row is one line), with word boxes rescaled to the
    TAT-DQA coordinate frame. Returns (doc, scales) where scales[p] maps back to PDF points."""
    import fitz

    pdf = fitz.open(path)
    pages, scales, next_id = [], [], 0
    for page in list(pdf)[:max_pages]:
        s = TARGET_WIDTH / page.rect.width
        # PDF text objects often hold one table cell each, so rows are rebuilt from geometry:
        # words whose vertical centres lie within half a line height share a row.
        words = sorted(((x0 * s, y0 * s, x1 * s, y1 * s, w) for x0, y0, x1, y1, w, *_ in page.get_text("words")),
                       key=lambda w: (w[1] + w[3]) / 2)
        rows = []
        for w in words:
            yc, hgt = (w[1] + w[3]) / 2, w[3] - w[1]
            if rows and abs(yc - rows[-1]["yc"]) < 0.5 * max(hgt, rows[-1]["h"]):
                rows[-1]["words"].append(w)
            else:
                rows.append({"yc": yc, "h": hgt, "words": [w]})
        blocks = []
        for row in rows:
            words = sorted(row["words"], key=lambda w: w[0])
            blocks.append({"id": next_id, "words": [w[4] for w in words], "wboxes": [list(w[:4]) for w in words],
                           "bbox": [min(w[0] for w in words), min(w[1] for w in words),
                                    max(w[2] for w in words), max(w[3] for w in words)]})
            next_id += 1
        pages.append({"size": [TARGET_WIDTH, page.rect.height * s], "blocks": blocks})
        scales.append(s)
    return {"pages": pages}, scales


@torch.no_grad()
def answer(model, tok, doc, question, max_new_tokens=400):
    prompt = tatllm.build_input(tok, doc, question)
    enc = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
    raw = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    parsed = tatllm.parse(raw)
    ans, scale = tatllm.to_prediction(parsed, use_executor=True)
    evidence, found = tatllm.ground(doc, parsed)
    return {"answer": ans, "scale": scale, "parsed": parsed, "evidence": evidence, "grounded": found, "raw": raw}


def highlight_pages(path, doc, scales, evidence, zoom=1.5):
    """PIL images of the PDF pages with cited blocks boxed in orange."""
    import fitz
    from PIL import Image, ImageDraw

    pdf = fitz.open(path)
    images, ev = [], set(evidence)
    for p_idx, (page, s) in enumerate(zip(list(pdf)[:len(doc["pages"])], scales)):
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        draw = ImageDraw.Draw(img, "RGBA")
        k = zoom / s
        for b in doc["pages"][p_idx]["blocks"]:
            if b["id"] in ev:
                x0, y0, x1, y1 = (v * k for v in b["bbox"])
                draw.rectangle([x0 - 3, y0 - 2, x1 + 3, y1 + 2], fill=(255, 165, 0, 60), outline=(230, 120, 0, 255), width=3)
        images.append(img)
    return images


def explain_html(result, doc):
    p = result["parsed"]
    by_id = {b["id"]: b for pg in doc["pages"] for b in pg["blocks"]}
    ev_lines = "".join(f"<li><code>[{i}] {html.escape(block_text(by_id[i]))}</code></li>" for i in result["evidence"] if i in by_id)
    how = (f"<b>Equation (computed by Python):</b> <code>{html.escape(p['equation'])}</code>"
           if p["type"] == "arithmetic" else f"<b>Evidence values:</b> {html.escape(' # '.join(p['evidence_text']))}")
    ans = ", ".join(result["answer"]) or "(no valid answer)"
    check = "✅ all evidence values found in the document" if result["grounded"] else "⚠️ some evidence values were not found in the document"
    return (f"<h3>Answer: {html.escape(ans)} {html.escape(result['scale'])}</h3>"
            f"<p><b>Question type:</b> {html.escape(p['type'] or 'unknown')}<br/>{how}</p>"
            f"<p><b>Evidence rows (highlighted on the page):</b></p><ul>{ev_lines or '<li>none found</li>'}</ul>"
            f"<p>{check}</p>")
