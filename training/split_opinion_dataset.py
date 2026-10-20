import json
from pathlib import Path

ENTITY_DIR = Path("data/labeled/entity")
OPINION_FILE = Path("data/labeled/opinion_spans.jsonl")
OUTPUT_DIR = Path("data/labeled/opinion")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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
    print(f"\n===== {split.upper()} =====")
    print("Comments:", len(records))
    print(
        "Opinion spans:",
        sum(len(r["opinions"]) for r in records)
    )
    print(
        "Without opinion:",
        sum(not r["opinions"] for r in records)
    )


def main():
    opinion_records = load_jsonl(OPINION_FILE)

    opinion_by_id = {}

    for record in opinion_records:
        key = str(record["comment_id"])

        if key in opinion_by_id:
            raise ValueError(
                f"Duplicate opinion comment_id: {key}"
            )

        opinion_by_id[key] = record

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

            if key not in opinion_by_id:
                raise ValueError(
                    f"Missing opinion annotation: {key}"
                )

            opinion_record = opinion_by_id[key]

            if entity_record["text"] != opinion_record["text"]:
                raise ValueError(
                    f"Text mismatch for comment_id: {key}"
                )

            output_records.append(opinion_record)
            used_ids.add(key)

        split_records[split] = output_records

    # 2. Tìm comment Opinion chưa thuộc split nào
    unused_ids = set(opinion_by_id) - used_ids

    new_records = [
        record
        for record in opinion_records
        if str(record["comment_id"]) in unused_ids
    ]

    print("\n===== NEW OPINION COMMENTS =====")
    print("New comments:", len(new_records))
    print(
        "New opinion spans:",
        sum(len(r["opinions"]) for r in new_records)
    )

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

    print("\nOpinion dataset split completed!")
    print(
        f"Added {len(new_records)} new comments to TRAIN."
    )


if __name__ == "__main__":
    main()