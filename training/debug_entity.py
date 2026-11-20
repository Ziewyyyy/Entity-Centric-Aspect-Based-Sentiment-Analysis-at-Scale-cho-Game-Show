
import sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

from train_entity import (
    load_jsonl,
    encode_record,
    tokenizer,
    id2label
)

DATA_DIR = ROOT / "data/labeled/entity"


def inspect_dataset(filename):
    path = DATA_DIR / filename
    records = load_jsonl(path)

    entity_counts = Counter()
    bio_counts = Counter()

    print(f"\n===== {filename.upper()} =====")
    print("Comments:", len(records))

    for record in records:
        for entity in record["entities"]:
            entity_counts[entity["type"]] += 1

        encoded = encode_record(record)

        for label_id in encoded["labels"]:
            if label_id != -100:
                bio_counts[id2label[label_id]] += 1

    print("Entity spans:", dict(entity_counts))
    print("BIO labels:", dict(bio_counts))


def show_examples():
    records = load_jsonl(DATA_DIR / "train.jsonl")

    examples = [
        r for r in records if r["entities"]
    ][:5]

    for record in examples:
        print("\n" + "=" * 60)
        print("Comment:", record["text"])
        print("Entities:", record["entities"])

        encoded = encode_record(record)

        tokens = tokenizer.convert_ids_to_tokens(
            encoded["input_ids"]
        )

        for token, label_id in zip(
            tokens,
            encoded["labels"]
        ):
            if label_id != -100:
                print(
                    f"{token:25} "
                    f"{id2label[label_id]}"
                )


if __name__ == "__main__":
    for filename in [
        "train.jsonl",
        "val.jsonl",
        "test.jsonl"
    ]:
        inspect_dataset(filename)

    show_examples()
