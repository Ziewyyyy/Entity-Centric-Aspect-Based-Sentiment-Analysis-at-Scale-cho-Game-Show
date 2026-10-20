
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Chỉ export comment từ 2 video mới
VIDEO_IDS = [
    "qKTqFeRb43w"
]

input_path = (
    ROOT / "data" / "cleaned" / "performance_cleaned.jsonl"
)

output_path = (
    ROOT / "data" / "labeled" / "label_studio_tasks_new.json"
)

tasks = []
seen_ids = set()

with open(input_path, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue

        item = json.loads(line)

        video_id = str(item.get("video_id", "")).strip()

        # Chỉ lấy comment của 2 video mới
        if video_id not in VIDEO_IDS:
            continue

        comment_id = str(item["comment_id"])

        # Loại trùng comment ID
        if comment_id in seen_ids:
            continue

        seen_ids.add(comment_id)

        tasks.append({
            "data": {
                "comment_id": comment_id,
                "video_id": video_id,
                "text": item["text"],
                "video_type": "performance"
            }
        })
        
        if len(tasks) >= 100:
            break

output_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(
        tasks,
        f,
        ensure_ascii=False,
        indent=2
    )

print(f"Exported {len(tasks)} new video tasks")
print(f"Saved to: {output_path}")
