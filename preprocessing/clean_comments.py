
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = ROOT / "data" / "raw" / "performance"
OUTPUT_FILE = ROOT / "data" / "cleaned" / "performance_cleaned.jsonl"


def clean_text(text):
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    total = 0
    valid = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as output:
        for path in INPUT_DIR.glob("*.jsonl"):
            with open(path, encoding="utf-8") as file:
                for line in file:
                    total += 1

                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    comment_id = item.get("comment_id")
                    text = clean_text(item.get("text", ""))

                    if not comment_id:
                        continue

                    if comment_id in seen:
                        continue

                    if len(text) < 3:
                        continue

                    seen.add(comment_id)

                    item["text"] = text

                    output.write(
                        json.dumps(item, ensure_ascii=False) + "\n"
                    )

                    valid += 1

    print(f"Raw comments: {total}")
    print(f"Cleaned comments: {valid}")
    print(f"Removed: {total - valid}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
