
import inspect
import json
import random
import re
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "labeled" / "entity"
OUTPUT_DIR = ROOT / "models" / "entity_phobert"

MODEL_NAME = "vinai/phobert-base-v2"

MAX_LENGTH = 256
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 1
EPOCHS = 5
LEARNING_RATE = 2e-5
SEED = 42

ENTITY_TYPES = [
    "ARTIST",
    "TEAM",
    "PERFORMANCE",
    "SHOW",
]

LABELS = ["O"] + [
    f"{prefix}-{name}"
    for name in ENTITY_TYPES
    for prefix in ("B", "I")
]

label2id = {
    name: idx
    for idx, name in enumerate(LABELS)
}

id2label = {
    idx: name
    for name, idx in label2id.items()
}


# ============================================================
# GPU
# ============================================================

def setup_gpu():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available.\n"
            "Please install CUDA-enabled PyTorch.\n"
            "Check with: python -c "
            "\"import torch; print(torch.cuda.is_available())\""
        )

    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(0)

    print("\n" + "=" * 60)
    print("GPU CONFIGURATION")
    print("=" * 60)
    print("Device:", device)
    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__)
    print("CUDA:", torch.version.cuda)
    print(f"VRAM: {props.total_memory / 1024**3:.2f} GB")
    print("FP16: Enabled")
    print("=" * 60)

    return device


# ============================================================
# SEED
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=False,
)


# ============================================================
# LOAD DATA
# ============================================================

def load_jsonl(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    records = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}: {e}"
                ) from e

    return records


# ============================================================
# VALIDATE ENTITY
# ============================================================

def validate_entities(record):
    text = record["text"]

    entities = sorted(
        record.get("entities", []),
        key=lambda e: (e["start"], e["end"]),
    )

    previous_end = 0

    for entity in entities:
        start = entity["start"]
        end = entity["end"]

        if entity["type"] not in ENTITY_TYPES:
            raise ValueError(
                f"Unknown entity type: {entity['type']}"
            )

        if not 0 <= start < end <= len(text):
            raise ValueError(
                f"Invalid offsets: {start}, {end}"
            )

        if text[start:end] != entity["text"]:
            raise ValueError(
                f"Entity span mismatch: "
                f"{entity['text']!r} != {text[start:end]!r}"
            )

        if start < previous_end:
            raise ValueError(
                "Overlapping entities are not supported"
            )

        previous_end = end

    return entities


# ============================================================
# ENTITY DICTIONARY
# ============================================================

def load_glued_aliases():
    path = ROOT / "annotation" / "entity_dictionary.json"

    if not path.exists():
        raise FileNotFoundError(
            f"Entity dictionary not found: {path}"
        )

    data = json.loads(
        path.read_text(encoding="utf-8")
    )

    aliases = set()

    for item in data.get("artists", []):
        names = [
            item.get("canonical_name", ""),
            *item.get("aliases", []),
        ]

        for alias in names:
            alias = alias.strip()

            if (
                len(alias) >= 4
                and re.fullmatch(
                    r"\w+",
                    alias,
                    flags=re.UNICODE,
                )
            ):
                aliases.add(alias.casefold())

    return tuple(
        sorted(
            aliases,
            key=lambda a: (-len(a), a),
        )
    )


GLUED_ALIASES = load_glued_aliases()


# ============================================================
# SPLIT GLUED ARTIST NAMES
# ============================================================

def split_glued_artist_names(token):
    folded = token.casefold()

    if len(folded) != len(token):
        return [token]

    memo = {}

    def solve(pos):
        if pos == len(folded):
            return []

        if pos in memo:
            return memo[pos]

        for alias in GLUED_ALIASES:
            if folded.startswith(alias, pos):
                tail = solve(pos + len(alias))

                if tail is not None:
                    memo[pos] = [
                        token[pos:pos + len(alias)],
                        *tail,
                    ]
                    return memo[pos]

        memo[pos] = None
        return None

    parts = solve(0)

    if parts is not None and len(parts) >= 2:
        return parts

    return [token]


# ============================================================
# TOKENIZATION WITH OFFSETS
# ============================================================

def segment_with_offsets(text):
    words = []

    for match in re.finditer(
        r"\w+|[^\w\s]",
        text,
        re.UNICODE,
    ):
        token = match.group()
        offset = match.start()

        parts = split_glued_artist_names(token)

        for part in parts:
            words.append({
                "text": part,
                "start": offset,
                "end": offset + len(part),
            })

            offset += len(part)

    return words


def encode_words(text):
    words = segment_with_offsets(text)

    input_ids = [tokenizer.cls_token_id]
    word_positions = []

    for word in words:
        prefix = " " if len(input_ids) > 1 else ""

        ids = tokenizer.encode(
            prefix + word["text"],
            add_special_tokens=False,
        )

        if not ids:
            raise ValueError(
                f"Cannot tokenize word: {word!r}"
            )

        word_positions.append(len(input_ids))
        input_ids.extend(ids)

    input_ids.append(tokenizer.sep_token_id)

    if len(input_ids) > MAX_LENGTH:
        raise ValueError(
            f"Sequence length {len(input_ids)} "
            f"exceeds MAX_LENGTH={MAX_LENGTH}"
        )

    return {
        "words": words,
        "input_ids": input_ids,
        "word_positions": word_positions,
    }


# ============================================================
# BIO LABELS
# ============================================================

def get_word_labels(words, entities):
    labels = []
    previous_entity_index = None

    for word in words:
        matched_index = None

        for i, entity in enumerate(entities):
            if (
                word["start"] < entity["end"]
                and word["end"] > entity["start"]
            ):
                if (
                    word["start"] < entity["start"]
                    or word["end"] > entity["end"]
                ):
                    raise ValueError(
                        "Entity boundary falls inside a token: "
                        f"token={word!r}, entity={entity!r}"
                    )

                matched_index = i
                break

        if matched_index is None:
            labels.append("O")
            previous_entity_index = None
        else:
            entity_type = entities[matched_index]["type"]

            tag = (
                "I"
                if matched_index == previous_entity_index
                else "B"
            )

            labels.append(
                f"{tag}-{entity_type}"
            )

            previous_entity_index = matched_index

    return labels


# ============================================================
# ENCODE RECORD
# ============================================================

def encode_record(record):
    entities = validate_entities(record)

    encoded = encode_words(
        record["text"]
    )

    word_labels = get_word_labels(
        encoded["words"],
        entities,
    )

    labels = [-100] * len(encoded["input_ids"])

    for position, tag in zip(
        encoded["word_positions"],
        word_labels,
    ):
        labels[position] = label2id[tag]

    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": [1] * len(encoded["input_ids"]),
        "labels": labels,
    }


def build_dataset(path):
    records = load_jsonl(path)

    if not records:
        raise ValueError(
            f"Empty dataset: {path}"
        )

    encoded = []

    for record in records:
        try:
            encoded.append(
                encode_record(record)
            )
        except Exception as e:
            raise ValueError(
                f"Error processing comment "
                f"{record.get('comment_id')}: {e}"
            ) from e

    return Dataset.from_list(encoded)


# ============================================================
# DECODE PREDICTIONS
# ============================================================

def decode_predictions(logits, labels):
    predictions = np.argmax(
        logits,
        axis=-1,
    )

    true_predictions = []
    true_labels = []

    for pred_row, gold_row in zip(
        predictions,
        labels,
    ):
        pred_tags = []
        gold_tags = []

        for pred_id, gold_id in zip(
            pred_row,
            gold_row,
        ):
            if gold_id == -100:
                continue

            pred_tags.append(
                id2label[int(pred_id)]
            )

            gold_tags.append(
                id2label[int(gold_id)]
            )

        true_predictions.append(pred_tags)
        true_labels.append(gold_tags)

    return true_predictions, true_labels


# ============================================================
# METRICS
# ============================================================

def compute_metrics(prediction):
    pred_tags, gold_tags = decode_predictions(
        prediction.predictions,
        prediction.label_ids,
    )

    return {
        "precision": precision_score(
            gold_tags,
            pred_tags,
            zero_division=0,
        ),
        "recall": recall_score(
            gold_tags,
            pred_tags,
            zero_division=0,
        ),
        "f1": f1_score(
            gold_tags,
            pred_tags,
            zero_division=0,
        ),
    }


# ============================================================
# MAIN TRAINING
# ============================================================

def main():
    set_seed(SEED)

    # GPU check
    device = setup_gpu()

    print("\n===== LOADING DATASETS =====")

    train_dataset = build_dataset(
        DATA_DIR / "train.jsonl"
    )

    val_dataset = build_dataset(
        DATA_DIR / "val.jsonl"
    )

    test_dataset = build_dataset(
        DATA_DIR / "test.jsonl"
    )

    print(f"Train: {len(train_dataset)}")
    print(f"Validation: {len(val_dataset)}")
    print(f"Test: {len(test_dataset)}")

    print("\n===== LABELS =====")

    for label, index in label2id.items():
        print(f"{index}: {label}")

    print("\n===== LOADING PHOBERT =====")

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id,
    )

    # Explicitly move model to GPU.
    model = model.to(device)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # TRAINING ARGUMENTS
    # ========================================================

    args_kwargs = {
        "output_dir": str(OUTPUT_DIR),

        "num_train_epochs": EPOCHS,

        "per_device_train_batch_size": BATCH_SIZE,
        "per_device_eval_batch_size": BATCH_SIZE,

        "gradient_accumulation_steps":
            GRADIENT_ACCUMULATION_STEPS,

        "learning_rate": LEARNING_RATE,
        "weight_decay": 0.01,

        "save_strategy": "epoch",
        "logging_steps": 10,

        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
        "greater_is_better": True,

        "save_total_limit": 2,
        "report_to": "none",
        "seed": SEED,

        # GPU acceleration
        "use_cpu": False,
        "fp16": True,
        "dataloader_pin_memory": True,

        # Optimization
        "optim": "adamw_torch",
    }

    params = inspect.signature(
        TrainingArguments.__init__
    ).parameters

    strategy_name = (
        "eval_strategy"
        if "eval_strategy" in params
        else "evaluation_strategy"
    )

    args_kwargs[strategy_name] = "epoch"

    training_args = TrainingArguments(
        **args_kwargs
    )

    # ========================================================
    # TRAINER
    # ========================================================

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,

        "data_collator":
            DataCollatorForTokenClassification(
                tokenizer=tokenizer
            ),

        "compute_metrics": compute_metrics,
    }

    trainer_params = inspect.signature(
        Trainer.__init__
    ).parameters

    tokenizer_arg = (
        "processing_class"
        if "processing_class" in trainer_params
        else "tokenizer"
    )

    trainer_kwargs[tokenizer_arg] = tokenizer

    trainer = Trainer(
        **trainer_kwargs
    )

    print("\n===== TRAINING DEVICE =====")
    print("Trainer device:", trainer.args.device)

    if trainer.args.device.type != "cuda":
        raise RuntimeError(
            "Trainer is not using CUDA."
        )

    print("\n===== START TRAINING =====")

    trainer.train()

    # ========================================================
    # VALIDATION
    # ========================================================

    print("\n===== VALIDATION =====")

    val_result = trainer.evaluate()
    print(val_result)

    # ========================================================
    # TEST
    # ========================================================

    print("\n===== TEST EVALUATION =====")

    test_result = trainer.predict(
        test_dataset
    )

    print(test_result.metrics)

    pred_tags, gold_tags = decode_predictions(
        test_result.predictions,
        test_result.label_ids,
    )

    print("\n===== CLASSIFICATION REPORT =====")

    print(
        classification_report(
            gold_tags,
            pred_tags,
            zero_division=0,
        )
    )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    print("\n===== SAVING MODEL =====")

    trainer.save_model(
        str(OUTPUT_DIR)
    )

    tokenizer.save_pretrained(
        str(OUTPUT_DIR)
    )

    (OUTPUT_DIR / "test_metrics.json").write_text(
        json.dumps(
            test_result.metrics,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    (OUTPUT_DIR / "labels.json").write_text(
        json.dumps(
            {
                "labels": LABELS,
                "label2id": label2id,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n===== TRAINING COMPLETED =====")
    print("GPU:", torch.cuda.get_device_name(0))
    print("Model saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
