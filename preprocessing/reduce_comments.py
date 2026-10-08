
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data/raw/performance"
BACKUP_DIR = ROOT / "data/backup/performance"

LIMIT = 100

BACKUP_DIR.mkdir(parents=True, exist_ok=True)

for file_path in RAW_DIR.glob("*.jsonl"):
    with open(file_path, "r", encoding="utf-8") as f:
        comments = [
            json.loads(line)
            for line in f
            if line.strip()
        ]

    backup_path = BACKUP_DIR / file_path.name

    if not backup_path.exists():
        backup_path.write_bytes(file_path.read_bytes())

    selected = comments[:LIMIT]

    with open(file_path, "w", encoding="utf-8") as f:
        for comment in selected:
            f.write(
                json.dumps(comment, ensure_ascii=False) + "\n"
            )

    print(
        f"{file_path.name}: "
        f"{len(comments)} -> {len(selected)} comments"
    )
