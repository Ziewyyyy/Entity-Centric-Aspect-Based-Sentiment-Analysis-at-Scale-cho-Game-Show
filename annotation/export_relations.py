"""Xuất quan hệ HAS_OPINION (Entity -> Opinion) từ Label Studio.

Cách chạy (từ thư mục gốc project):
    python annotation/export_relations.py
    python annotation/export_relations.py --input data/labeled/label_studio_export.json

Mỗi dòng JSONL là một comment:
{
  "comment_id": "...", "video_id": "...", "video_type": "...", "text": "...",
  "entities":  [{"text", "start", "end", "type"}],     # sắp xếp theo vị trí
  "opinions":  [{"text", "start", "end"}],             # sắp xếp theo vị trí
  "relations": [{"entity": i, "opinion": j}]           # chỉ số trong 2 danh sách trên
}
"""

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT = ROOT / "data" / "labeled" / "label_studio_export_merged.json"
OUTPUT_PATH = ROOT / "data" / "labeled" / "relation_spans.jsonl"

ENTITY_TYPES = {"ARTIST", "TEAM", "PERFORMANCE", "SHOW"}
OPINION_LABEL = "OPINION"
RELATION_LABEL = "HAS_OPINION"


# ============================================================
# NORMALIZE EXPORTED SPAN TEXT
# ============================================================

def normalize_span(text):
    """Chuẩn hóa whitespace khi so sánh span (giống export_opinion_spans.py)."""
    if not isinstance(text, str):
        return text

    return (
        text.replace("\\r\\n", "\n")
            .replace("\\n", "\n")
            .replace("\\r", "\r")
            .replace("\\t", "\t")
            .strip()
    )


# ============================================================
# CLEAN ONE SPAN
# ============================================================

def clean_span(task_id, text, value):
    """Kiểm tra offset, đối chiếu text xuất ra, cắt khoảng trắng hai đầu."""
    start = value.get("start")
    end = value.get("end")

    if (
        not isinstance(start, int)
        or isinstance(start, bool)
        or not isinstance(end, int)
        or isinstance(end, bool)
        or not 0 <= start < end <= len(text)
    ):
        raise ValueError(f"Invalid offsets in task {task_id}: {value}")

    span_text = text[start:end]
    exported_text = value.get("text")

    if exported_text is not None and exported_text != span_text:
        if normalize_span(exported_text) == normalize_span(span_text):
            print(
                f"[WARNING] Task {task_id}: span text differs only by "
                "whitespace. Using original text offsets."
            )
        else:
            raise ValueError(
                f"Offset mismatch in task {task_id}: "
                f"exported={exported_text!r}, actual={span_text!r}"
            )

    left_trim = len(span_text) - len(span_text.lstrip())
    right_trim = len(span_text) - len(span_text.rstrip())

    new_start = start + left_trim
    new_end = end - right_trim

    if new_start >= new_end:
        raise ValueError(f"Empty span after trimming in task {task_id}")

    return {
        "text": text[new_start:new_end],
        "start": new_start,
        "end": new_end,
    }


def check_overlap(task_id, spans, name):
    for previous, current in zip(spans, spans[1:]):
        if current["start"] < previous["end"]:
            raise ValueError(
                f"Overlapping {name} in task {task_id}: {previous} and {current}"
            )


# ============================================================
# EXTRACT ONE TASK
# ============================================================

def extract_relations(task, stats):
    task_id = task.get("id")
    data = task.get("data", {})
    text = data.get("text", "")

    if not isinstance(text, str):
        raise ValueError(f"Task {task_id}: invalid text")

    valid_annotations = [
        annotation
        for annotation in task.get("annotations", [])
        if not annotation.get("was_cancelled", False)
        and annotation.get("result") is not None
    ]

    if not valid_annotations:
        return None

    if len(valid_annotations) > 1:
        raise ValueError(
            f"Task {task_id} has multiple annotations. "
            "Choose an annotation policy before exporting."
        )

    results = valid_annotations[0].get("result", [])

    # --------------------------------------------------------
    # 1. Đọc vùng Entity / Opinion
    # --------------------------------------------------------

    entities = []   # (region_id, span)
    opinions = []

    for result in results:
        if result.get("type") != "labels":
            continue

        value = result.get("value", {})
        labels = value.get("labels", [])

        if OPINION_LABEL in labels:
            opinions.append((result["id"], clean_span(task_id, text, value)))
            continue

        entity_types = [label for label in labels if label in ENTITY_TYPES]

        if entity_types:
            span = clean_span(task_id, text, value)
            span["type"] = entity_types[0]
            entities.append((result["id"], span))

    entities.sort(key=lambda item: (item[1]["start"], item[1]["end"]))
    opinions.sort(key=lambda item: (item[1]["start"], item[1]["end"]))

    check_overlap(task_id, [s for _, s in entities], "entities")
    check_overlap(task_id, [s for _, s in opinions], "opinions")

    entity_index = {region_id: i for i, (region_id, _) in enumerate(entities)}
    opinion_index = {region_id: i for i, (region_id, _) in enumerate(opinions)}

    # --------------------------------------------------------
    # 2. Đọc quan hệ
    # --------------------------------------------------------

    relations = []
    seen = set()

    for result in results:
        if result.get("type") != "relation":
            continue

        labels = result.get("labels", [])

        if labels and RELATION_LABEL not in labels:
            stats["other_relation_label"] += 1
            continue

        if not labels:
            # Project chỉ có một loại quan hệ nên coi quan hệ thiếu nhãn là HAS_OPINION.
            stats["unlabeled_relation"] += 1
            print(
                f"[WARNING] Task {task_id}: relation without label "
                f"treated as {RELATION_LABEL}."
            )

        source = result.get("from_id")
        target = result.get("to_id")

        # Gán ngược chiều (Opinion -> Entity): đảo lại cho đúng.
        if source in opinion_index and target in entity_index:
            source, target = target, source
            stats["reversed_relation"] += 1
            print(f"[WARNING] Task {task_id}: reversed relation, swapped to Entity -> Opinion.")

        if source not in entity_index or target not in opinion_index:
            stats["skipped_relation"] += 1
            print(
                f"[WARNING] Task {task_id}: relation is not Entity -> Opinion "
                f"(from={source}, to={target}). Skipped."
            )
            continue

        pair = (entity_index[source], opinion_index[target])

        if pair in seen:
            stats["duplicate_relation"] += 1
            continue

        seen.add(pair)
        relations.append({"entity": pair[0], "opinion": pair[1]})

    relations.sort(key=lambda r: (r["opinion"], r["entity"]))

    # --------------------------------------------------------
    # 3. Thống kê
    # --------------------------------------------------------

    stats["entities"] += len(entities)
    stats["opinions"] += len(opinions)
    stats["relations"] += len(relations)

    linked = Counter(r["opinion"] for r in relations)
    stats["opinions_with_relation"] += len(linked)
    stats["opinions_multi_entity"] += sum(1 for c in linked.values() if c > 1)

    # --------------------------------------------------------
    # 4. Record
    # --------------------------------------------------------

    return {
        "comment_id": data.get("comment_id", str(task_id)),
        "video_id": data.get("video_id"),
        "video_type": data.get("video_type"),
        "text": text,
        "entities": [span for _, span in entities],
        "opinions": [span for _, span in opinions],
        "relations": relations,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Export HAS_OPINION relations")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"File not found: {args.input}")

    tasks = json.loads(args.input.read_text(encoding="utf-8"))

    if not isinstance(tasks, list):
        raise ValueError("Expected a JSON array of Label Studio tasks")

    stats = Counter()
    records = []
    skipped = 0

    for task in tasks:
        record = extract_relations(task, stats)

        if record is None:
            skipped += 1
            continue

        records.append(record)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    candidate_pairs = sum(len(r["entities"]) * len(r["opinions"]) for r in records)

    print("\n" + "=" * 60)
    print("RELATION EXPORT")
    print("=" * 60)
    print("Tasks:", len(tasks))
    print("Exported comments:", len(records))
    print("Skipped (no annotation):", skipped)
    print("Entities:", stats["entities"])
    print("Opinions:", stats["opinions"])
    print("HAS_OPINION relations:", stats["relations"])
    print("Opinions with relation:", stats["opinions_with_relation"])
    print("Opinions linked to >1 entity:", stats["opinions_multi_entity"])
    print("Candidate pairs (entity x opinion):", candidate_pairs)
    print("  positive:", stats["relations"])
    print("  negative:", candidate_pairs - stats["relations"])
    print("-" * 60)
    print("Relations without label (kept):", stats["unlabeled_relation"])
    print("Reversed relations (swapped):", stats["reversed_relation"])
    print("Skipped relations (not Entity->Opinion):", stats["skipped_relation"])
    print("Skipped relations (other label):", stats["other_relation_label"])
    print("Duplicate relations:", stats["duplicate_relation"])
    print("Saved to:", args.output)
    print("=" * 60)


if __name__ == "__main__":
    main()
