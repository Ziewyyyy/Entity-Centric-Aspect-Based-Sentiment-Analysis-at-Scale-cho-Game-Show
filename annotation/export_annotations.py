
import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data/labeled/label_studio_export.json"
OUTPUT = ROOT / "data/labeled/performance_labeled.jsonl"

ENTITY_TYPES = {
    "ARTIST", "TEAM", "SHOW", "PERFORMANCE"
}

ASPECTS = {
    "VOCAL", "DANCE", "PERFORMANCE", "VISUAL",
    "PERSONALITY", "INTERACTION", "CONTENT",
    "PRODUCTION", "EDITING", "POPULARITY", "AWARD_RESULT", "GENERAL"
}

SENTIMENTS = {"POS", "NEG", "NEU"}

stats = Counter()
errors = []


def extract_task(task):
    text = task["data"]["text"]
    annotations = []

    for annotation in task.get("annotations", []):
        if annotation.get("was_cancelled"):
            continue

        regions = {}
        relations = []

        for result in annotation.get("result", []):
            result_type = result.get("type")
            region_id = result.get("id")

            if result_type == "relation":
                relations.append(result)
                continue

            if not region_id:
                continue

            region = regions.setdefault(region_id, {})

            value = result.get("value", {})

            if result_type == "labels":
                region["start"] = value.get("start")
                region["end"] = value.get("end")
                region["text"] = value.get("text")
                region["label"] = (
                    value.get("labels") or [None]
                )[0]

            elif result_type == "choices":
                choice = (value.get("choices") or [None])[0]

                if result.get("from_name") == "aspect":
                    region["aspect"] = choice

                elif result.get("from_name") == "sentiment":
                    region["sentiment"] = choice

        for relation in relations:
            from_id = relation.get("from_id")
            to_id = relation.get("to_id")

            entity = regions.get(from_id)
            opinion = regions.get(to_id)

            if not entity or not opinion:
                errors.append(
                    f"Missing region in task {task['id']}"
                )
                continue

            if entity.get("label") not in ENTITY_TYPES:
                errors.append(
                    f"Invalid entity in task {task['id']}"
                )
                continue

            if opinion.get("label") != "OPINION":
                errors.append(
                    f"Invalid opinion in task {task['id']}"
                )
                continue

            aspect = opinion.get("aspect")
            sentiment = opinion.get("sentiment")

            if aspect not in ASPECTS:
                errors.append(
                    f"Missing/invalid aspect: task {task['id']}"
                )
                continue

            if sentiment not in SENTIMENTS:
                errors.append(
                    f"Missing/invalid sentiment: task {task['id']}"
                )
                continue

            entity_start = entity.get("start")
            entity_end = entity.get("end")
            opinion_start = opinion.get("start")
            opinion_end = opinion.get("end")

            offsets = [
                entity_start, entity_end,
                opinion_start, opinion_end
            ]

            if not all(isinstance(x, int) for x in offsets):
                errors.append(f"Invalid offsets: task {task['id']}")
                continue

            if not (
                0 <= entity_start < entity_end <= len(text)
                and 0 <= opinion_start < opinion_end <= len(text)
            ):
                errors.append(f"Offsets out of range: task {task['id']}")
                continue

            if (
                text[entity_start:entity_end] != entity.get("text")
                or text[opinion_start:opinion_end] != opinion.get("text")
            ):
                errors.append(f"Span mismatch: task {task['id']}")
                continue

            labels = relation.get("labels", [])

            if labels and "HAS_OPINION" not in labels:
                errors.append(
                    f"Unexpected relation: task {task['id']}"
                )
                continue

            annotations.append({
                "entity": entity["text"],
                "entity_type": entity["label"],
                "entity_start": entity_start,
                "entity_end": entity_end,
                "aspect": aspect,
                "opinion_span": opinion["text"],
                "opinion_start": opinion_start,
                "opinion_end": opinion_end,
                "sentiment": sentiment,
                "relation": "HAS_OPINION"
            })

    return {
        "comment_id": task["data"]["comment_id"],
        "video_id": task["data"]["video_id"],
        "video_type": task["data"].get("video_type"),
        "text": text,
        "annotations": annotations
    }


def main():
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    with open(INPUT, encoding="utf-8") as f:
        tasks = json.load(f)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT, "w", encoding="utf-8") as f:
        for task in tasks:
            # Only export submitted, non-cancelled tasks
            if not any(
                not a.get("was_cancelled")
                for a in task.get("annotations", [])
            ):
                continue

            record = extract_task(task)

            stats["comments"] += 1
            stats["annotations"] += len(record["annotations"])

            for ann in record["annotations"]:
                stats[ann["sentiment"]] += 1

            f.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    print("\n===== EXPORT SUMMARY =====")
    print(f"Comments: {stats['comments']}")
    print(f"ABSA annotations: {stats['annotations']}")
    print(f"POS: {stats['POS']}")
    print(f"NEG: {stats['NEG']}")
    print(f"NEU: {stats['NEU']}")
    print(f"Errors: {len(errors)}")

    for error in errors[:20]:
        print("WARNING:", error)

    print(f"\nSaved to: {OUTPUT}")


if __name__ == "__main__":
    main()
