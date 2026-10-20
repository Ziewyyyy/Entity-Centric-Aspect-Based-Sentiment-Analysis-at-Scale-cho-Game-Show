
import json
import argparse
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT = ROOT / "data" / "labeled" / "label_studio_export.json"
DEFAULT_OUTPUT = ROOT / "data" / "labeled" / "sentiment_spans.jsonl"

SENTIMENTS = {"POS", "NEG", "NEU"}

ASPECTS = {
    "GENERAL", "VOCAL", "CONTENT", "VISUAL",
    "POPULARITY", "PRODUCTION", "PERFORMANCE",
    "DANCE", "INTERACTION", "PERSONALITY",
    "AWARD_RESULT", "EDITING",
}


def get_choices(result):
    value = result.get("value", {})
    return value.get("choices", [])


def export_sentiment(input_path, output_path):
    with open(input_path, encoding="utf-8") as f:
        tasks = json.load(f)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = Counter()
    records = []

    for task in tasks:
        data = task.get("data", {})
        text = data.get("text", "")
        comment_id = data.get("comment_id", str(task.get("id")))

        annotations = [
            a for a in task.get("annotations", [])
            if not a.get("was_cancelled", False)
        ]

        if not annotations:
            stats["no_annotation"] += 1
            continue

        # Chọn annotation mới nhất nếu có nhiều annotation.
        annotation = max(
            annotations,
            key=lambda a: a.get("updated_at") or ""
        )

        results = annotation.get("result", [])

        opinion_regions = {}
        aspects = {}
        sentiments = {}

        for result in results:
            region_id = result.get("id")
            value = result.get("value", {})
            result_type = result.get("type")

            if result_type == "labels":
                labels = value.get("labels", [])

                if "OPINION" in labels:
                    opinion_regions[region_id] = {
                        "start": value.get("start"),
                        "end": value.get("end"),
                        "text": value.get("text", ""),
                    }

            elif result_type == "choices":
                choices = get_choices(result)

                for choice in choices:
                    if choice in ASPECTS:
                        aspects[region_id] = choice

                    if choice in SENTIMENTS:
                        sentiments[region_id] = choice

        for region_id, opinion in opinion_regions.items():
            start = opinion["start"]
            end = opinion["end"]

            if not isinstance(start, int) or not isinstance(end, int):
                stats["invalid_offset"] += 1
                continue

            if not (0 <= start < end <= len(text)):
                stats["invalid_offset"] += 1
                continue

            opinion_text = text[start:end]
            
            # Khai báo trước vòng lặp qua tasks
            missing_records = []

            # Trong nhánh thiếu Sentiment
            if region_id not in sentiments:
                missing_records.append({
                    "comment_id": comment_id,
                    "text": text,
                    "opinion": opinion_text,
                    "start": start,
                    "end": end,
                    "aspect": aspects.get(region_id),
                })
                stats["missing_sentiment"] += 1
                continue

            if region_id not in sentiments:
                stats["missing_sentiment"] += 1
                continue

            if region_id not in aspects:
                stats["missing_aspect"] += 1
                continue

            record = {
                "comment_id": comment_id,
                "video_id": data.get("video_id"),
                "video_type": data.get("video_type"),
                "text": text,
                "opinion": opinion_text,
                "start": start,
                "end": end,
                "aspect": aspects[region_id],
                "sentiment": sentiments[region_id],
            }

            records.append(record)
            stats["exported"] += 1
            stats[sentiments[region_id]] += 1

    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Xuất danh sách Opinion thiếu Sentiment
    missing_file = (
        ROOT / "data" / "analysis" / "sentiment"
        / "missing_sentiment.jsonl"
    )

    missing_file.parent.mkdir(parents=True, exist_ok=True)

    with open(missing_file, "w", encoding="utf-8") as f:
        for row in missing_records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Missing sentiment file: {missing_file}")

    print("\n===== SENTIMENT EXPORT =====")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Total tasks: {len(tasks)}")
    print(f"Exported spans: {stats['exported']}")
    print(f"POS: {stats['POS']}")
    print(f"NEG: {stats['NEG']}")
    print(f"NEU: {stats['NEU']}")
    print(f"Missing sentiment: {stats['missing_sentiment']}")
    print(f"Missing aspect: {stats['missing_aspect']}")
    print(f"Invalid offsets: {stats['invalid_offset']}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    args = parser.parse_args()
    export_sentiment(args.input, args.output)


if __name__ == "__main__":
    main()
