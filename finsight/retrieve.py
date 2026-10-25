"""Retrieval-augmented generation, no training.

1. Example RAG: for each question, the most similar *solved* training questions are shown to the
   model as worked examples (their evidence rows, unit rows, gold program and scale). This teaches
   the dataset's conventions (sign of a change, x100 for percentages, which unit an answer carries,
   span vs. multi-span) without changing any weights. Similarity mixes the question's words with its
   *template* (years and numbers masked), so "change in X between 2018 and 2019" also matches other
   change questions, not only questions about X.
2. Row RAG: BM25 over a document's rows, to find the rows a query is about (risk report, long
   documents, highlighting).

Only the training split is ever retrieved from; dev and test answers are never shown to the model.
"""
import math
import re
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from .data import block_text
from .prompt import gold_target

UNIT_ROW = re.compile(r"thousand|million|billion|'000|\$000|in \$|%|percent", re.I)
YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
TOKEN = re.compile(r"[a-z0-9]+")


def template(question):
    """Question with years and numbers masked: keeps the operation, drops the specifics."""
    q = YEAR.sub(" yyyy ", question.lower())
    return NUMBER.sub(" nnn ", q)


UNIT_WORDS = (("billion", re.compile(r"\bbillions?\b|\$\s*bn\b", re.I)),
              ("million", re.compile(r"\bmillions?\b|\$\s*m\b|'000,000", re.I)),
              ("thousand", re.compile(r"\bthousands?\b|'000|\$000|\$\s*k\b", re.I)))


def doc_unit(doc):
    """The money unit the document states most often ("thousand", "million", "billion" or "")."""
    text = " ".join(" ".join(b["words"]) for p in doc["pages"] for b in p["blocks"])
    counts = {u: len(rx.findall(text)) for u, rx in UNIT_WORDS}
    best = max(counts, key=counts.get)
    return best if counts[best] else ""


def example_context(doc, evidence, max_unit_rows=3):
    """Gold evidence rows + a few unit-stating rows, in document order, with original ids."""
    blocks = [b for p in doc["pages"] for b in p["blocks"]]
    keep = set(evidence)
    for b in blocks:
        if len(keep) >= len(evidence) + max_unit_rows:
            break
        text = block_text(b)
        if b["id"] not in keep and UNIT_ROW.search(text) and len(text) < 160:
            keep.add(b["id"])
    return "\n".join(f"[{b['id']}] {block_text(b)}" for b in blocks if b["id"] in keep)


class ExampleRetriever:
    """k most similar solved training questions as (context, question, answer) examples."""

    # Defaults chosen on dev with scripts/retrieval_check.py: template only (weight 1.0) and a 0.2 bonus
    # for the same money unit give the most examples with the dev question's type, scale and calculation.
    def __init__(self, train_docs, train_questions, template_weight=1.0, unit_bonus=0.2):
        self.items = [(q, t) for q in train_questions if q["evidence"] and (t := gold_target(q)) is not None]
        self.docs = train_docs
        self.w, self.unit_bonus = template_weight, unit_bonus
        units = {uid: doc_unit(d) for uid, d in train_docs.items()}
        self.units = np.array([units[q["doc_uid"]] for q, _ in self.items])
        texts = [q["question"] for q, _ in self.items]
        self.word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        self.word_m = self.word.fit_transform(texts)
        self.tmpl = TfidfVectorizer(ngram_range=(1, 3), min_df=2, sublinear_tf=True)
        self.tmpl_m = self.tmpl.fit_transform([template(t) for t in texts])

    def scores(self, question_text, unit=None):
        word = linear_kernel(self.word.transform([question_text]), self.word_m).ravel()
        tmpl = linear_kernel(self.tmpl.transform([template(question_text)]), self.tmpl_m).ravel()
        s = (1 - self.w) * word + self.w * tmpl
        if unit is not None and self.unit_bonus:
            s = s + self.unit_bonus * (self.units == unit)
        return s

    def nearest(self, question, k=4, per_doc=2, doc=None):
        """The k most similar training questions (most similar first): no duplicate question texts,
        at most per_doc from one document, never the question itself. With doc, examples from
        documents that state the same money unit are preferred (unit_bonus)."""
        sims = self.scores(question["question"], doc_unit(doc) if doc is not None else None)
        out, seen, docs = [], set(), Counter()
        for i in sims.argsort()[::-1]:
            q, target = self.items[i]
            if q["uid"] == question.get("uid") or q["question"] in seen or docs[q["doc_uid"]] >= per_doc:
                continue
            seen.add(q["question"])
            docs[q["doc_uid"]] += 1
            out.append((q, target))
            if len(out) == k:
                break
        return out

    def __call__(self, question, k=4, doc=None):
        """Examples for prompt.build_messages, most similar last (closest to the query)."""
        return [(example_context(self.docs[q["doc_uid"]], q["evidence"]), q["question"], t)
                for q, t in self.nearest(question, k, doc=doc)][::-1]


class RowRetriever:
    """BM25 over the rows (blocks) of one document."""

    def __init__(self, doc, k1=1.2, b=0.75):
        self.blocks = [b for p in doc["pages"] for b in p["blocks"]]
        self.texts = [block_text(b) for b in self.blocks]
        self.toks = [TOKEN.findall(t.lower()) for t in self.texts]
        self.k1, self.b = k1, b
        self.avg = sum(map(len, self.toks)) / max(1, len(self.toks))
        df = Counter(w for toks in self.toks for w in set(toks))
        n = len(self.toks)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def __call__(self, query, k=5):
        """[(block id, text, score)] for the k best-matching rows, best first."""
        q = TOKEN.findall(query.lower())
        scored = []
        for blk, text, toks in zip(self.blocks, self.texts, self.toks):
            tf = Counter(toks)
            s = sum(self.idf.get(w, 0) * tf[w] * (self.k1 + 1) /
                    (tf[w] + self.k1 * (1 - self.b + self.b * len(toks) / self.avg)) for w in q if w in tf)
            if s > 0:
                scored.append((blk["id"], text, s))
        return sorted(scored, key=lambda x: -x[2])[:k]
