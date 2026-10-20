
"""Convert Label Studio export to aspect train/val/test JSONL.

Run from project root:
    python training/prepare_aspect_dataset.py --input data/labeled/label_studio_export.json

Reuses existing Entity comment_id splits when available.
Otherwise creates a seeded 70/15/15 comment-level split.

Prevents cross-split text leakage, prioritizing test > val > train.
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

ASPECTS = {
    "GENERAL", "VOCAL", "CONTENT", "VISUAL",
    "POPULARITY", "PRODUCTION", "PERFORMANCE",
    "DANCE", "INTERACTION", "PERSONALITY",
    "AWARD_RESULT", "EDITING"
}

SPLITS = ("train", "val", "test")


def load_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize_text(text):
    return " ".join(text.casefold().split())


def remove_cross_split_duplicates(out):
    """Remove text duplicates across splits.

    Priority: test > val > train.
    Keep all opinion samples from the same comment in the winning split.
    """
    seen_texts = set()
    removed = Counter()

    for split in ("test", "val", "train"):
        clean_records = []

        # Determine which normalized texts belong to this split.
        current_texts = {
            normalize_text(r["text"])
            for r in out[split]
        }

        # Remove records whose comment text is already in a higher-priority split.
        for record in out[split]:
            key = normalize_text(record["text"])

            if key in seen_texts:
                removed[split] += 1
                continue

            clean_records.append(record)

        out[split] = clean_records
        seen_texts.update(current_texts)

    return removed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    tasks = json.loads(args.input.read_text(encoding="utf-8"))
    root = args.root

    split_dir = root / "data/labeled/entity"
    existing = {}

    # 1. Reuse Entity splits when available.
    if all(
        (split_dir / f"{s}.jsonl").exists()
        for s in SPLITS
    ):
        for split in SPLITS:
            for rec in load_jsonl(split_dir / f"{split}.jsonl"):
                cid = str(rec["comment_id"])

                if cid in existing:
                    raise ValueError(
                        f"Overlapping split comment_id: {cid}"
                    )

                existing[cid] = split

        print("[INFO] Reusing Entity comment_id splits.")

    else:
        ids = sorted({
            str(t["data"].get("comment_id", t["id"]))
            for t in tasks
        })

        random.Random(args.seed).shuffle(ids)

        n = len(ids)
        n_train = round(n * 0.70)
        n_val = round(n * 0.15)

        existing = {
            cid: (
                "train"
                if i < n_train
                else "val"
                if i < n_train + n_val
                else "test"
            )
            for i, cid in enumerate(ids)
        }

        print(
            "[WARNING] Entity splits not found; "
            "creating new comment-level splits."
        )

    # 2. Extract aspect samples.
    out = {s: [] for s in SPLITS}
    issues = Counter()

    for task in tasks:
        cid = str(
            task["data"].get("comment_id", task["id"])
        )

        split = existing.get(cid)

        if split is None:
            split = "train"
            existing[cid] = split
            issues["new_comments_added_to_train"] += 1

        comment = task["data"]["text"]

        annotations = [
            a for a in task.get("annotations", [])
            if not a.get("was_cancelled")
        ]

        if not annotations:
            issues["no_submitted_annotation"] += 1
            continue

        results = annotations[-1].get("result", [])

        opinions = [
            r for r in results
            if r.get("type") == "labels"
            and "OPINION" in r.get("value", {}).get("labels", [])
        ]

        for op in opinions:
            v = op["value"]
            start = v.get("start")
            end = v.get("end")

            if (
                not isinstance(start, int)
                or not isinstance(end, int)
                or not (0 <= start < end <= len(comment))
            ):
                issues["invalid_opinion_offsets"] += 1
                continue

            if comment[start:end] != v.get("text"):
                issues["opinion_text_mismatch"] += 1
                continue

            labels = [
                r["value"].get("choices", [])
                for r in results
                if r.get("id") == op.get("id")
                and r.get("from_name") == "aspect"
                and r.get("type") == "choices"
            ]

            if len(labels) != 1 or len(labels[0]) != 1:
                issues["missing_or_ambiguous_aspect"] += 1
                continue

            aspect = labels[0][0]

            if aspect not in ASPECTS:
                issues["unknown_aspect"] += 1
                continue

            out[split].append({
                "comment_id": cid,
                "text": comment,
                "opinion": comment[start:end],
                "start": start,
                "end": end,
                "aspect": aspect
            })

    # 3. Remove cross-split duplicates.
    print("\n===== CROSS-SPLIT DEDUPLICATION =====")

    before = {
        split: len(out[split])
        for split in SPLITS
    }

    removed = remove_cross_split_duplicates(out)

    for split in SPLITS:
        print(
            f"{split}: "
            f"{before[split]} -> {len(out[split])} "
            f"(removed {removed[split]})"
        )

    # 4. Verify no text overlap.
    text_sets = {
        split: {
            normalize_text(r["text"])
            for r in out[split]
        }
        for split in SPLITS
    }

    for a, b in [
        ("train", "val"),
        ("train", "test"),
        ("val", "test")
    ]:
        overlap = text_sets[a] & text_sets[b]

        if overlap:
            raise ValueError(
                f"Data leakage detected: "
                f"{a}/{b}: {len(overlap)} duplicated texts"
            )

    print("[OK] No cross-split duplicate texts.")

    # 5. Save JSONL.
    target = root / "data/labeled/aspect"
    target.mkdir(parents=True, exist_ok=True)

    print("\n===== DATASET SUMMARY =====")

    for split, records in out.items():
        path = target / f"{split}.jsonl"

        with path.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(
                    json.dumps(rec, ensure_ascii=False) + "\n"
                )

        unique_comments = len({
            r["comment_id"] for r in records
        })

        print(
            f"{split}: {len(records)} opinion samples, "
            f"{unique_comments} comments -> {path}"
        )

        print(
            "  Aspect counts:",
            dict(Counter(r["aspect"] for r in records))
        )

    print("\nSkipped/issues:", dict(issues))
    print("Removed duplicate samples:", dict(removed))


if __name__ == "__main__":
    main()
