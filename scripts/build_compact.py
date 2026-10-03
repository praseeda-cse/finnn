"""Pack TAT-DQA (QA json + per-document layout json) into one small file per split.

Run locally (no GPU needed):
    python scripts/build_compact.py --dataset "C:/Users/praseeda/Desktop/fp/dataset" --out data

Output: data/tatdqa_{train,dev,test}.json.gz, each {"docs": {uid: doc}, "questions": [...]}
where doc = {"pages": [{"size": [w, h], "blocks": [{"id", "uuid", "bbox", "words", "wboxes"}]}]}.
Block ids are numbered across the whole document so the model can cite them.
"""
import argparse
import gzip
import json
import os

SPLITS = {
    "train": ("tatdqa_dataset_train (1).json", "tatdqa_docs_train/train"),
    "dev": ("tatdqa_dataset_dev (1).json", "tatdqa_docs_dev/dev"),
    "test": ("tatdqa_dataset_test_gold (1).json", "tatdqa_docs_test/test"),
}


def load_doc(path):
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    pages, next_id = [], 0
    for page in raw["pages"]:
        blocks = []
        for b in sorted(page["blocks"], key=lambda b: b["order"]):
            words = b["words"]["word_list"]
            if not words:
                continue
            blocks.append({
                "id": next_id,
                "uuid": b["uuid"],
                "bbox": b["bbox"],
                "words": words,
                "wboxes": b["words"]["bbox_list"],
            })
            next_id += 1
        pages.append({"size": page["bbox"][2:], "blocks": blocks})
    return {"pages": pages}


def build(dataset_dir, split, out_dir):
    qa_file, doc_dir = SPLITS[split]
    with open(os.path.join(dataset_dir, qa_file), encoding="utf-8") as f:
        qa = json.load(f)

    docs, questions, missing = {}, [], 0
    for entry in qa:
        uid = entry["doc"]["uid"]
        path = os.path.join(dataset_dir, doc_dir, uid + ".json")
        if not os.path.exists(path):
            missing += 1
            continue
        doc = load_doc(path)
        docs[uid] = doc
        uuid_to_id = {b["uuid"]: b["id"] for p in doc["pages"] for b in p["blocks"]}
        for q in entry["questions"]:
            evidence = sorted({uuid_to_id[u] for m in q.get("block_mapping", []) for u in m if u in uuid_to_id})
            questions.append({
                "uid": q["uid"],
                "doc_uid": uid,
                "question": q["question"],
                "answer": q["answer"],
                "answer_type": q["answer_type"],
                "scale": q["scale"],
                "derivation": q.get("derivation", ""),
                "facts": q.get("facts", []),
                "evidence": evidence,
            })

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"tatdqa_{split}.json.gz")
    with gzip.open(out, "wt", encoding="utf-8") as f:
        json.dump({"docs": docs, "questions": questions}, f)
    print(f"{split}: {len(docs)} docs, {len(questions)} questions, {missing} missing docs -> {out} "
          f"({os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default="data")
    args = ap.parse_args()
    for s in SPLITS:
        build(args.dataset, s, args.out)
