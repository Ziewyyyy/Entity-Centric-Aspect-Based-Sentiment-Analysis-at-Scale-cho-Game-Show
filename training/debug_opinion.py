
import json
from pathlib import Path
from collections import Counter

DATA_DIR = Path("data/labeled/opinion")


def inspect_dataset(split):
    path = DATA_DIR / f"{split}.jsonl"
    errors = []
    opinion_count = 0
    without_opinion = 0

    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue

            record = json.loads(line)
            text = record["text"]
            opinions = sorted(
                record.get("opinions", []),
                key=lambda x: (x["start"], x["end"])
            )

            if not opinions:
                without_opinion += 1

            opinion_count += len(opinions)
            previous_end = 0

            for opinion in opinions:
                start = opinion["start"]
                end = opinion["end"]
                span_text = opinion["text"]

                if not (
                    isinstance(start, int)
                    and isinstance(end, int)
                    and 0 <= start < end <= len(text)
                ):
                    errors.append((line_no, "Invalid offsets", opinion))
                    continue

                if text[start:end] != span_text:
                    errors.append((line_no, "Text mismatch", opinion))

                if start < previous_end:
                    errors.append((line_no, "Overlapping spans", opinion))

                previous_end = max(previous_end, end)

    print(f"\n===== {split.upper()} =====")
    print("Opinion spans:", opinion_count)
    print("Without opinion:", without_opinion)
    print("Errors:", len(errors))

    for error in errors[:10]:
        print(error)


for split in ["train", "val", "test"]:
    inspect_dataset(split)
