"""Load the compact TAT-DQA files and turn a document into model-readable text.

Each text block becomes one line: `[id] word word | cell | cell`. A ` | ` is inserted
where the horizontal gap between two words is large, which recovers table columns
from the PDF word boxes (every table row in TAT-DQA is a single block).
"""
import gzip
import json

COLUMN_GAP = 18          # px gap (on the ~1240px-wide page) that counts as a column break
NO_SEP_AFTER = {"$", "(", "€", "£", "¥", "-$", "($"}
NO_SEP_BEFORE = {")", "%", ")%", "%)", ",", "."}


def load_split(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    return data["docs"], data["questions"]


def block_text(block):
    words, boxes = block["words"], block["wboxes"]
    out = words[0]
    for prev, cur, pbox, cbox in zip(words, words[1:], boxes, boxes[1:]):
        gap = cbox[0] - pbox[2]
        if gap > COLUMN_GAP and prev not in NO_SEP_AFTER and cur not in NO_SEP_BEFORE:
            out += " | " + cur
        else:
            out += " " + cur
    return out


def render_doc(doc):
    lines = []
    multi = len(doc["pages"]) > 1
    for i, page in enumerate(doc["pages"], 1):
        if multi:
            lines.append(f"--- page {i} ---")
        for b in page["blocks"]:
            lines.append(f"[{b['id']}] {block_text(b)}")
    return "\n".join(lines)
