import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_opinion import (
    load_jsonl,
    tokenize_with_offsets,
    create_word_labels,
)

DATA_DIR = Path("data/labeled/opinion")


def analyze_split(split):
    path = DATA_DIR / f"{split}.jsonl"
    records = load_jsonl(path)

    label_counts = Counter()
    total_opinions = 0
    comments_with_opinion = 0

    for record in records:
        text = record["text"]
        opinions = record.get("opinions", [])

        words = tokenize_with_offsets(text)
        labels = create_word_labels(words, opinions, text)

        label_counts.update(labels)
        total_opinions += len(opinions)

        if opinions:
            comments_with_opinion += 1

    total_tokens = sum(label_counts.values())

    print(f"\n===== {split.upper()} =====")
    print(f"Total comments: {len(records)}")
    print(f"Comments with opinion: {comments_with_opinion}")
    print(f"Comments without opinion: {len(records) - comments_with_opinion}")
    print(f"Total opinion spans: {total_opinions}")
    print(f"Total tokens: {total_tokens}")

    print("\nLABEL DISTRIBUTION")
    print(f"{'Label':<15} {'Count':>10} {'Percentage':>12}")

    for label in ["O", "B-OPINION", "I-OPINION"]:
        count = label_counts[label]
        percentage = count / total_tokens * 100 if total_tokens else 0

        print(f"{label:<15} {count:>10} {percentage:>11.2f}%")

    positive = label_counts["B-OPINION"] + label_counts["I-OPINION"]
    negative = label_counts["O"]

    print("\nCLASS BALANCE")
    print(f"Opinion tokens: {positive}")
    print(f"Non-opinion tokens: {negative}")
    print(f"O / Opinion ratio: {negative / positive:.2f}:1" if positive else "No opinion tokens")


if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        analyze_split(split)