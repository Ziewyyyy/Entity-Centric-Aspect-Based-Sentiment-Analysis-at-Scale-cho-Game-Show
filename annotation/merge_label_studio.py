
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "labeled"

OLD_FILE = DATA_DIR / "label_studio_export.json"
NEW_FILE = DATA_DIR / "project-12-at-2026-10-20-15-25-52de28c9.json"
OUTPUT_FILE = DATA_DIR / "label_studio_export_merged.json"

with open(OLD_FILE, encoding="utf-8") as f:
    old_tasks = json.load(f)

with open(NEW_FILE, encoding="utf-8") as f:
    new_tasks = json.load(f)

merged = []
seen = set()

for task in old_tasks + new_tasks:
    comment_id = task.get("data", {}).get("comment_id")

    if not comment_id or comment_id in seen:
        continue

    seen.add(comment_id)
    merged.append(task)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(merged, f, ensure_ascii=False, indent=2)

print("Old:", len(old_tasks))
print("New:", len(new_tasks))
print("Merged:", len(merged))
print("Saved:", OUTPUT_FILE)
