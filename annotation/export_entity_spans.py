
import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data/labeled/label_studio_export.json"
OUTPUT = ROOT / "data/labeled/entity_spans.jsonl"

ENTITY_TYPES = {
    "ARTIST",
    "TEAM",
    "PERFORMANCE",
    "SHOW"
}


def extract_entities(task):
    data = task.get("data", {})

    text = data.get("text", "")
    comment_id = data.get("comment_id")
    video_id = data.get("video_id")

    entities = []
    seen = set()

    annotations = task.get("annotations", [])

    # Chỉ dùng annotation mới nhất đã Submit,
    # tránh trộn các phiên bản annotation.
    valid_annotations = [
        ann for ann in annotations
        if not ann.get("was_cancelled", False)
        and ann.get("result") is not None
    ]

    if not valid_annotations:
        return None

    annotation = max(
        valid_annotations,
        key=lambda a: (
            a.get("updated_at") or "",
            a.get("id") or 0
        )
    )

    for result in annotation.get("result", []):

        if result.get("type") != "labels":
            continue

        value = result.get("value", {})

        labels = value.get("labels", [])

        if not labels:
            continue

        entity_type = labels[0]

        # Bỏ qua OPINION và các nhãn khác
        if entity_type not in ENTITY_TYPES:
            continue

        start = value.get("start")
        end = value.get("end")
        entity_text = value.get("text", "")

        if not isinstance(start, int) or not isinstance(end, int):
            print(f"[WARN] Invalid offsets: {comment_id}")
            continue

        if not (0 <= start < end <= len(text)):
            print(f"[WARN] Out-of-range span: {comment_id}")
            continue

        actual_text = text[start:end]

        if actual_text != entity_text:
            print(
                f"[WARN] Span mismatch: {comment_id} "
                f"expected={entity_text!r}, actual={actual_text!r}"
            )
            continue

        key = (start, end, entity_type)

        if key in seen:
            continue

        seen.add(key)

        entities.append({
            "text": entity_text,
            "start": start,
            "end": end,
            "type": entity_type
        })

    entities.sort(key=lambda x: (x["start"], x["end"]))

    return {
        "comment_id": comment_id,
        "video_id": video_id,
        "video_type": data.get("video_type", "performance"),
        "text": text,
        "entities": entities
    }


def main():
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Cannot find Label Studio export: {INPUT}"
        )

    with open(INPUT, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    if not isinstance(tasks, list):
        raise ValueError("Expected Label Studio JSON array")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    records = []
    stats = Counter()
    seen_comments = set()

    for task in tasks:
        record = extract_entities(task)

        if record is None:
            stats["not_submitted"] += 1
            continue

        comment_id = record["comment_id"]

        if comment_id is not None:
            if comment_id in seen_comments:
                stats["duplicate_comments"] += 1
                print(f"[WARN] Duplicate comment: {comment_id}")
                continue

            seen_comments.add(comment_id)

        records.append(record)
        stats["comments"] += 1

        if record["entities"]:
            stats["comments_with_entities"] += 1
        else:
            stats["comments_without_entities"] += 1

        for entity in record["entities"]:
            stats[entity["type"]] += 1
            stats["total_entities"] += 1

    with open(OUTPUT, "w", encoding="utf-8") as f:
        for record in records:
            f.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    print("\n===== ENTITY EXPORT SUMMARY =====")
    print(f"Comments exported: {stats['comments']}")
    print(f"Comments with entities: {stats['comments_with_entities']}")
    print(f"Comments without entities: {stats['comments_without_entities']}")
    print(f"Total entity spans: {stats['total_entities']}")

    for entity_type in sorted(ENTITY_TYPES):
        print(f"{entity_type}: {stats[entity_type]}")

    print(f"Not submitted: {stats['not_submitted']}")
    print(f"Duplicate comments: {stats['duplicate_comments']}")
    print(f"\nSaved to: {OUTPUT}")


if __name__ == "__main__":
    main()
