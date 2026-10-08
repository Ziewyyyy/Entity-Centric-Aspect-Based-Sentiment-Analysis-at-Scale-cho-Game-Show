
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

input_path = ROOT / "data/cleaned/performance_cleaned.jsonl"
output_path = ROOT / "data/labeled/label_studio_tasks.json"

tasks = []

with open(input_path, encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)

        tasks.append({
            "data": {
                "comment_id": item["comment_id"],
                "video_id": item["video_id"],
                "text": item["text"],
                "video_type": "performance"
            }
        })

output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(tasks, f, ensure_ascii=False, indent=2)

print(f"Exported {len(tasks)} tasks")
