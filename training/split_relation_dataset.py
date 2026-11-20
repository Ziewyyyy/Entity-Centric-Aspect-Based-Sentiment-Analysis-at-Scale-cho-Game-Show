"""Chia dataset Relation theo đúng split của Entity (train/val/test).

Cách chạy (từ thư mục gốc project), sau export_relations.py:
    python annotation/split_relation_dataset.py

Giữ nguyên train/val/test của Entity để cả 5 model dùng chung một tập test,
tránh rò rỉ dữ liệu. Comment chưa thuộc split nào sẽ được thêm vào TRAIN.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ENTITY_DIR = ROOT / "data" / "labeled" / "entity"
RELATION_FILE = ROOT / "data" / "labeled" / "relation_spans.jsonl"
OUTPUT_DIR = ROOT / "data" / "labeled" / "relation"


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def save_jsonl(path, records):
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )


def print_statistics(split, records):
    pairs = sum(len(r["entities"]) * len(r["opinions"]) for r in records)
    positives = sum(len(r["relations"]) for r in records)

    print(f"\n===== {split.upper()} =====")
    print("Comments:", len(records))
    print("Entities:", sum(len(r["entities"]) for r in records))
    print("Opinions:", sum(len(r["opinions"]) for r in records))
    print("Candidate pairs:", pairs)
    print("  positive (HAS_OPINION):", positives)
    print("  negative:", pairs - positives)
    print(
        "Comments with >1 entity:",
        sum(len(r["entities"]) > 1 for r in records),
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    relation_records = load_jsonl(RELATION_FILE)

    relation_by_id = {}

    for record in relation_records:
        key = str(record["comment_id"])

        if key in relation_by_id:
            raise ValueError(
                f"Duplicate relation comment_id: {key}"
            )

        relation_by_id[key] = record

    used_ids = set()
    split_records = {}

    # 1. Giữ nguyên các split từ Entity
    for split in ["train", "val", "test"]:
        entity_records = load_jsonl(
            ENTITY_DIR / f"{split}.jsonl"
        )

        output_records = []

        for entity_record in entity_records:
            key = str(entity_record["comment_id"])

            if key in used_ids:
                raise ValueError(
                    f"Comment appears in multiple splits: {key}"
                )

            if key not in relation_by_id:
                raise ValueError(
                    f"Missing relation annotation: {key}"
                )

            relation_record = relation_by_id[key]

            if entity_record["text"] != relation_record["text"]:
                raise ValueError(
                    f"Text mismatch for comment_id: {key}"
                )

            output_records.append(relation_record)
            used_ids.add(key)

        split_records[split] = output_records

    # 2. Tìm comment Relation chưa thuộc split nào
    unused_ids = set(relation_by_id) - used_ids

    new_records = [
        record
        for record in relation_records
        if str(record["comment_id"]) in unused_ids
    ]

    print("\n===== NEW RELATION COMMENTS =====")
    print("New comments:", len(new_records))

    # 3. Thêm comment mới vào Train
    split_records["train"].extend(new_records)

    # 4. Kiểm tra không trùng ID giữa các split
    all_ids = []

    for records in split_records.values():
        all_ids.extend(
            str(r["comment_id"]) for r in records
        )

    if len(all_ids) != len(set(all_ids)):
        raise ValueError(
            "Duplicate comment_id across splits!"
        )

    # 5. Lưu dữ liệu
    for split in ["train", "val", "test"]:
        output_path = OUTPUT_DIR / f"{split}.jsonl"

        save_jsonl(
            output_path,
            split_records[split]
        )

        print_statistics(
            split,
            split_records[split]
        )

    print("\nRelation dataset split completed!")
    print(
        f"Added {len(new_records)} new comments to TRAIN."
    )


if __name__ == "__main__":
    main()
