
import json
import random
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data/labeled/entity_spans.jsonl"
OUTPUT_DIR = ROOT / "data/labeled/entity"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42

ENTITY_TYPES = {
    "ARTIST",
    "TEAM",
    "PERFORMANCE",
    "SHOW"
}


def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON at line {line_number}: {e}"
                )

            records.append(record)

    return records


def validate_records(records):
    seen_ids = set()

    for index, record in enumerate(records):
        comment_id = record.get("comment_id")
        text = record.get("text")
        entities = record.get("entities")

        if comment_id is None or str(comment_id).strip() == "":
            raise ValueError(
                f"Missing comment_id at record {index}"
            )

        comment_id = str(comment_id)

        if comment_id in seen_ids:
            raise ValueError(
                f"Duplicate comment_id: {comment_id}"
            )

        seen_ids.add(comment_id)

        if not isinstance(text, str):
            raise ValueError(
                f"Invalid text: {comment_id}"
            )

        if not isinstance(entities, list):
            raise ValueError(
                f"Invalid entities: {comment_id}"
            )

        for entity in entities:
            start = entity.get("start")
            end = entity.get("end")
            entity_text = entity.get("text")
            entity_type = entity.get("type")

            if entity_type not in ENTITY_TYPES:
                raise ValueError(
                    f"Unknown entity type: {entity_type}"
                )

            if not isinstance(start, int) or not isinstance(end, int):
                raise ValueError(
                    f"Invalid offsets: {comment_id}"
                )

            if not (0 <= start < end <= len(text)):
                raise ValueError(
                    f"Out-of-range span: {comment_id}"
                )

            if text[start:end] != entity_text:
                raise ValueError(
                    f"Span mismatch: {comment_id}, "
                    f"expected={entity_text!r}, "
                    f"actual={text[start:end]!r}"
                )

    print(f"[OK] Validated {len(records)} records")


def save_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )


def print_statistics(name, records):
    entity_counts = Counter()

    for record in records:
        for entity in record["entities"]:
            entity_counts[entity["type"]] += 1

    with_entities = sum(
        bool(record["entities"]) for record in records
    )

    print(f"\n===== {name.upper()} =====")
    print(f"Comments: {len(records)}")
    print(f"Comments with entities: {with_entities}")
    print(f"Comments without entities: {len(records) - with_entities}")

    for entity_type in sorted(ENTITY_TYPES):
        print(f"{entity_type}: {entity_counts[entity_type]}")


def main():
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT}"
        )

    if abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) > 1e-9:
        raise ValueError("Split ratios must sum to 1")

    records = load_jsonl(INPUT)

    if len(records) < 3:
        raise ValueError(
            "Need at least 3 comments to create 3 splits"
        )

    validate_records(records)

    rng = random.Random(SEED)
    rng.shuffle(records)

    total = len(records)

    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_data = records[:train_end]
    val_data = records[train_end:val_end]
    test_data = records[val_end:]

    if not train_data or not val_data or not test_data:
        raise ValueError("One of the splits is empty")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    save_jsonl(OUTPUT_DIR / "train.jsonl", train_data)
    save_jsonl(OUTPUT_DIR / "val.jsonl", val_data)
    save_jsonl(OUTPUT_DIR / "test.jsonl", test_data)

    # Kiểm tra không có comment xuất hiện ở nhiều tập
    train_ids = {str(r["comment_id"]) for r in train_data}
    val_ids = {str(r["comment_id"]) for r in val_data}
    test_ids = {str(r["comment_id"]) for r in test_data}

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    print("\n===== SPLIT SUMMARY =====")
    print(f"Total comments: {total}")
    print(f"Seed: {SEED}")

    print_statistics("Train", train_data)
    print_statistics("Validation", val_data)
    print_statistics("Test", test_data)

    print("\n[OK] No overlapping comment IDs")
    print(f"[OK] Saved datasets to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
