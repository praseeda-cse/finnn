"""In-context examples retrieved from the training set (no training needed).

For each question, the k most similar training questions (TF-IDF over the question text) are
shown to the model as solved examples. Each example contains only its evidence rows plus the
rows that state units ("in thousands", "%"), so the model sees how the dataset's answer type and
scale conventions apply, while the prompt stays short."""
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from .data import block_text
from .prompt import gold_target

UNIT_ROW = re.compile(r"thousand|million|billion|'000|\$000|in \$|%|percent", re.I)


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
    def __init__(self, train_docs, train_questions):
        self.items = []
        for q in train_questions:
            target = gold_target(q)
            if target is not None and q["evidence"]:
                self.items.append((q, target))
        self.docs = train_docs
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        self.matrix = self.vec.fit_transform([q["question"] for q, _ in self.items])

    def __call__(self, question, k=4):
        """Returns k (context, question, answer) examples, most similar last (closest to the query)."""
        sims = linear_kernel(self.vec.transform([question["question"]]), self.matrix).ravel()
        out, seen = [], set()
        for i in sims.argsort()[::-1]:
            q, target = self.items[i]
            if q["question"] in seen or q["uid"] == question.get("uid"):
                continue
            seen.add(q["question"])
            out.append((example_context(self.docs[q["doc_uid"]], q["evidence"]), q["question"], target))
            if len(out) == k:
                break
        return out[::-1]
