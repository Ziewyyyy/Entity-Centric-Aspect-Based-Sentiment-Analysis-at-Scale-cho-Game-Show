
"""
Split Sentiment dataset using Opinion reference splits.

Input:
    data/labeled/sentiment_spans.jsonl

Reference:
    data/labeled/opinion/train.jsonl
    data/labeled/opinion/val.jsonl
    data/labeled/opinion/test.jsonl

Output:
    data/labeled/sentiment/train.jsonl
    data/labeled/sentiment/val.jsonl
    data/labeled/sentiment/test.jsonl
"""

import json
import random
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "data/labeled/sentiment_spans.jsonl"
REFERENCE_DIR = ROOT / "data/labeled/opinion"
OUTPUT_DIR = ROOT / "data/labeled/sentiment"
ANALYSIS_DIR = ROOT / "data/analysis/sentiment"

SPLITS = ("train", "val", "test")
PRIORITY = ("test", "val", "train")
VALID_SENTIMENTS = {"POS", "NEG", "NEU"}
SEED = 42


def load_jsonl(path):
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy: {path}")

    rows = []

    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"JSON lỗi tại {path}, dòng {line_number}: {e}"
                ) from e

    return rows


def save_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(text):
    return " ".join(text.casefold().split())


def validate_sentiment_records(records):
    print("\n===== VALIDATING SENTIMENT DATA =====")

    required = (
        "comment_id", "text", "opinion",
        "start", "end", "aspect", "sentiment"
    )

    for index, row in enumerate(records, 1):
        missing = [key for key in required if key not in row]

        if missing:
            raise ValueError(
                f"Record {index} thiếu trường: {missing}"
            )

        if not isinstance(row["comment_id"], str) or not row["comment_id"]:
            raise ValueError(f"Record {index}: comment_id không hợp lệ")

        if not isinstance(row["text"], str):
            raise ValueError(f"Record {index}: text không hợp lệ")

        if row["sentiment"] not in VALID_SENTIMENTS:
            raise ValueError(
                f"Record {index}: sentiment không hợp lệ"
            )

        text = row["text"]
        opinion = row["opinion"]
        start = row["start"]
        end = row["end"]

        if (
            type(start) is not int
            or type(end) is not int
            or not isinstance(opinion, str)
        ):
            raise ValueError(
                f"Record {index}: start/end/opinion không hợp lệ"
            )

        if not (0 <= start < end <= len(text)):
            raise ValueError(
                f"Record {index}: invalid offsets [{start}:{end}]"
            )

        if text[start:end] != opinion:
            raise ValueError(
                f"Record {index}: opinion không khớp text\n"
                f"Expected: {text[start:end]!r}\n"
                f"Actual: {opinion!r}"
            )

    print("Valid records:", len(records))
    return records


def load_reference_splits():
    print("\n===== LOADING REFERENCE SPLITS =====")

    split_ids = {}
    reference_texts = defaultdict(set)

    for split in SPLITS:
        rows = load_jsonl(REFERENCE_DIR / f"{split}.jsonl")

        ids = set()

        for row in rows:
            cid = row["comment_id"]
            ids.add(cid)

            text = row.get("text")
            if isinstance(text, str) and text.strip():
                reference_texts[normalize_text(text)].add(split)

        split_ids[split] = ids

        print(
            f"{split.upper()}: "
            f"{len(rows)} rows, "
            f"{len(ids)} unique comments"
        )

    return split_ids, reference_texts


def check_comment_id_leakage(split_ids):
    print("\n===== CHECKING COMMENT ID LEAKAGE =====")

    for i, split_a in enumerate(SPLITS):
        for split_b in SPLITS[i + 1:]:
            overlap = split_ids[split_a] & split_ids[split_b]

            if overlap:
                raise ValueError(
                    f"Comment ID leakage: {split_a}/{split_b}: "
                    f"{len(overlap)} IDs"
                )

    print("No comment ID leakage detected.")


def assign_records_to_splits(records, split_ids, reference_texts):
    print("\n===== ASSIGNING RECORDS =====")

    outputs = {split: [] for split in SPLITS}

    id_to_split = {
        cid: split
        for split in SPLITS
        for cid in split_ids[split]
    }

    by_id = defaultdict(list)

    for row in records:
        by_id[row["comment_id"]].append(row)

    # Một comment ID không được chứa nhiều văn bản khác nhau.
    for cid, rows in by_id.items():
        texts = {normalize_text(r["text"]) for r in rows}

        if len(texts) != 1:
            raise ValueError(
                f"Comment {cid} có nhiều nội dung khác nhau"
            )

    new_by_text = defaultdict(list)

    for cid, rows in by_id.items():
        if cid in id_to_split:
            outputs[id_to_split[cid]].extend(rows)
        else:
            key = normalize_text(rows[0]["text"])
            new_by_text[key].append(cid)

    new_ids = sum(len(ids) for ids in new_by_text.values())

    print("New comment IDs:", new_ids)

    # Gom những văn bản mới theo nội dung chuẩn hóa.
    new_texts = sorted(new_by_text.keys())

    rng = random.Random(SEED)
    rng.shuffle(new_texts)

    new_counts = Counter()
    unmatched_groups = []

    for key in new_texts:
        matches = reference_texts.get(key, set())

        if matches:
            # Nếu trùng reference, dùng split đã có.
            target = next(
                split for split in PRIORITY if split in matches
            )

            for cid in new_by_text[key]:
                outputs[target].extend(by_id[cid])
                new_counts[target] += 1
        else:
            unmatched_groups.append(key)

    # Chia các nhóm chưa có trong reference theo 70/15/15.
    total = len(unmatched_groups)

    train_end = int(total * 0.70)
    val_end = int(total * 0.85)

    for index, key in enumerate(unmatched_groups):
        if index < train_end:
            target = "train"
        elif index < val_end:
            target = "val"
        else:
            target = "test"

        for cid in new_by_text[key]:
            outputs[target].extend(by_id[cid])
            new_counts[target] += 1

    print("New comment distribution:", dict(new_counts))

    for split in SPLITS:
        print(
            f"{split.upper()}: "
            f"{len(outputs[split])} sentiment spans"
        )

    return outputs


def find_cross_split_duplicates(outputs):
    print("\n===== CROSS-SPLIT TEXT LEAKAGE =====")

    locations = defaultdict(list)

    for split in SPLITS:
        for row in outputs[split]:
            key = normalize_text(row["text"])

            locations[key].append({
                "split": split,
                "comment_id": row["comment_id"],
                "text": row["text"],
                "opinion": row["opinion"],
                "sentiment": row["sentiment"]
            })

    duplicates = {}

    for text, items in locations.items():
        involved = {item["split"] for item in items}

        if len(involved) > 1:
            duplicates[text] = items

    print("Duplicated text groups:", len(duplicates))

    for key, items in list(sorted(duplicates.items()))[:10]:
        print("\nText:", repr(key))

        for item in items:
            print(
                f"  {item['split'].upper()}: "
                f"{item['comment_id']}"
            )

    return duplicates


def remove_cross_split_duplicates(outputs):
    print("\n===== REMOVING CROSS-SPLIT DUPLICATES =====")

    seen_texts = set()
    removed_records = []

    for split in PRIORITY:
        original = outputs[split]
        retained = []

        # Giữ tất cả spans trong cùng một split.
        for row in original:
            key = normalize_text(row["text"])

            if key in seen_texts:
                removed_records.append({
                    "removed_from": split,
                    **row
                })
            else:
                retained.append(row)

        seen_texts.update(
            normalize_text(row["text"]) for row in retained
        )

        outputs[split] = retained

        print(
            f"{split.upper()}: "
            f"{len(original)} -> {len(retained)} spans"
        )

    print("Total removed spans:", len(removed_records))

    return outputs, removed_records


def validate_final_splits(outputs):
    print("\n===== FINAL LEAKAGE CHECK =====")

    id_sets = {}
    text_sets = {}

    for split in SPLITS:
        id_sets[split] = {
            row["comment_id"] for row in outputs[split]
        }

        text_sets[split] = {
            normalize_text(row["text"]) for row in outputs[split]
        }

    for i, split_a in enumerate(SPLITS):
        for split_b in SPLITS[i + 1:]:
            if id_sets[split_a] & id_sets[split_b]:
                raise ValueError(
                    f"Comment ID leakage: {split_a}/{split_b}"
                )

            if text_sets[split_a] & text_sets[split_b]:
                raise ValueError(
                    f"Text leakage: {split_a}/{split_b}"
                )

    print("No comment ID leakage.")
    print("No text leakage.")


def print_statistics(outputs):
    print("\n" + "=" * 60)
    print("FINAL DATASET STATISTICS")
    print("=" * 60)

    total_spans = 0

    for split in SPLITS:
        rows = outputs[split]

        sentiment_counts = Counter(
            row["sentiment"] for row in rows
        )

        unique_comments = {
            row["comment_id"] for row in rows
        }

        total_spans += len(rows)

        print(f"\n===== {split.upper()} =====")
        print("Samples:", len(rows))
        print("Comments:", len(unique_comments))
        print("Sentiment distribution:")

        for sentiment in ("POS", "NEG", "NEU"):
            count = sentiment_counts[sentiment]

            percentage = (
                count / len(rows) * 100 if rows else 0
            )

            print(
                f"  {sentiment}: "
                f"{count} ({percentage:.2f}%)"
            )

    print("\nTotal final spans:", total_spans)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SENTIMENT DATASET SPLIT")
    print("=" * 60)

    print("\nInput:", INPUT_FILE)
    print("Reference:", REFERENCE_DIR)
    print("Output:", OUTPUT_DIR)

    # 1. Load sentiment data.
    records = load_jsonl(INPUT_FILE)
    records = validate_sentiment_records(records)

    # 2. Load Opinion reference splits.
    split_ids, reference_texts = load_reference_splits()

    # 3. Validate reference IDs.
    check_comment_id_leakage(split_ids)

    # 4. Assign records.
    outputs = assign_records_to_splits(
        records, split_ids, reference_texts
    )

    # 5. Detect duplicate texts.
    duplicates = find_cross_split_duplicates(outputs)

    duplicate_report = (
        ANALYSIS_DIR / "duplicate_text_report.json"
    )

    with open(duplicate_report, "w", encoding="utf-8") as f:
        json.dump(
            duplicates, f, ensure_ascii=False, indent=2
        )

    # 6. Remove duplicate texts across Sentiment splits.
    outputs, removed_records = remove_cross_split_duplicates(
        outputs
    )

    removed_file = ANALYSIS_DIR / "removed_duplicates.jsonl"
    save_jsonl(removed_file, removed_records)

    # 7. Validate final Sentiment splits.
    validate_final_splits(outputs)

    # 8. Save final datasets.
    for split in SPLITS:
        save_jsonl(
            OUTPUT_DIR / f"{split}.jsonl",
            outputs[split]
        )

    # 9. Print statistics.
    print_statistics(outputs)

    print("\n" + "=" * 60)
    print("SPLIT COMPLETED")
    print("=" * 60)

    print("\nOutput files:")

    for split in SPLITS:
        print(OUTPUT_DIR / f"{split}.jsonl")

    print("\nAnalysis files:")
    print(duplicate_report)
    print(removed_file)


if __name__ == "__main__":
    main()
