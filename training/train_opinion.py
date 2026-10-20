
import json
import random
import re
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)
from seqeval.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_NAME = "vinai/phobert-base-v2"

DATA_DIR = ROOT / "data" / "labeled" / "opinion"
OUTPUT_DIR = Path("D:/AI_Models/opinion_phobert")

LABELS = [
    "O",
    "B-OPINION",
    "I-OPINION",
]

label2id = {
    label: i
    for i, label in enumerate(LABELS)
}

id2label = {
    i: label
    for label, i in label2id.items()
}

MAX_LENGTH = 256
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 1
EPOCHS = 5
LEARNING_RATE = 2e-5
SEED = 42

TOKEN_PATTERN = re.compile(
    r"\w+|[^\w\s]",
    re.UNICODE,
)


# ============================================================
# GPU CONFIGURATION
# ============================================================

def setup_gpu():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA khong kha dung!\n"
            "PyTorch chua nhan GPU NVIDIA.\n"
            "Kiem tra bang lenh:\n"
            'python -c "import torch; '
            'print(torch.cuda.is_available())"'
        )

    device = torch.device("cuda:0")

    torch.cuda.set_device(device)

    props = torch.cuda.get_device_properties(0)

    print("\n" + "=" * 60)
    print("GPU CONFIGURATION")
    print("=" * 60)

    print("Device:", device)
    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__)
    print("CUDA:", torch.version.cuda)

    print(
        f"VRAM: {props.total_memory / 1024**3:.2f} GB"
    )

    print("Mixed Precision: FP16")
    print("Batch Size:", BATCH_SIZE)

    print("=" * 60)

    return device


# ============================================================
# RANDOM SEED
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# LOAD DATASET
# ============================================================

def load_jsonl(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    records = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
                records.append(record)

            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON at "
                    f"{path}:{line_number}: {e}"
                ) from e

    if not records:
        raise ValueError(
            f"Empty dataset: {path}"
        )

    return records


# ============================================================
# TOKENIZATION WITH OFFSETS
# ============================================================

def tokenize_with_offsets(text):
    return [
        {
            "text": match.group(),
            "start": match.start(),
            "end": match.end(),
        }
        for match in TOKEN_PATTERN.finditer(text)
    ]


# ============================================================
# CREATE BIO LABELS
# ============================================================

def create_word_labels(words, opinions, text):
    labels = ["O"] * len(words)

    for opinion in opinions:
        start = opinion["start"]
        end = opinion["end"]

        if not (
            isinstance(start, int)
            and isinstance(end, int)
            and 0 <= start < end <= len(text)
        ):
            raise ValueError(
                f"Invalid opinion offsets: {opinion}"
            )

        if text[start:end] != opinion["text"]:
            raise ValueError(
                f"Opinion offset mismatch: {opinion}"
            )

        indices = []

        for i, word in enumerate(words):
            ws = word["start"]
            we = word["end"]

            overlaps = (
                ws < end
                and we > start
            )

            if not overlaps:
                continue

            if ws < start or we > end:
                raise ValueError(
                    "Opinion boundary falls inside a token:\n"
                    f"Token: {word}\n"
                    f"Opinion: {opinion}\n"
                    f"Comment: {text}"
                )

            indices.append(i)

        if not indices:
            raise ValueError(
                f"No tokens found for opinion: {opinion}"
            )

        for j, index in enumerate(indices):
            if labels[index] != "O":
                raise ValueError(
                    f"Overlapping opinion labels: {opinion}"
                )

            labels[index] = (
                "B-OPINION"
                if j == 0
                else "I-OPINION"
            )

    return labels


# ============================================================
# PHOBERT TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=False,
)


# ============================================================
# ENCODE RECORD
# ============================================================

def encode_record(record):
    text = record["text"]
    opinions = record.get("opinions", [])

    words = tokenize_with_offsets(text)

    word_labels = create_word_labels(
        words,
        opinions,
        text,
    )

    input_ids = [
        tokenizer.cls_token_id
    ]

    labels = [-100]

    for word, label in zip(
        words,
        word_labels,
    ):
        token_ids = tokenizer.encode(
            word["text"],
            add_special_tokens=False,
        )

        if not token_ids:
            raise ValueError(
                f"Tokenizer returned no tokens: {word}"
            )

        input_ids.extend(token_ids)

        # Calculate loss only on first subword.
        labels.append(
            label2id[label]
        )

        labels.extend(
            [-100] * (len(token_ids) - 1)
        )

    input_ids.append(
        tokenizer.sep_token_id
    )

    labels.append(-100)

    if len(input_ids) > MAX_LENGTH:
        raise ValueError(
            f"Sequence too long: "
            f"{len(input_ids)} tokens. "
            f"Comment ID: {record.get('comment_id')}"
        )

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


# ============================================================
# PREPARE DATASET
# ============================================================

def prepare_dataset(records):
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

    for pred_row, label_row in zip(
        predictions,
        labels,
    ):
        pred_tags = []
        gold_tags = []

        for pred_id, gold_id in zip(
            pred_row,
            label_row,
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
# EVALUATION METRICS
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
# MAIN
# ============================================================

def main():
    set_seed(SEED)

    # Require NVIDIA CUDA GPU.
    device = setup_gpu()

    print("\n===== LOADING DATASETS =====")

    train_records = load_jsonl(
        DATA_DIR / "train.jsonl"
    )

    val_records = load_jsonl(
        DATA_DIR / "val.jsonl"
    )

    test_records = load_jsonl(
        DATA_DIR / "test.jsonl"
    )

    print("Train:", len(train_records))
    print("Validation:", len(val_records))
    print("Test:", len(test_records))

    print("\n===== ENCODING DATASETS =====")

    train_dataset = prepare_dataset(
        train_records
    )

    val_dataset = prepare_dataset(
        val_records
    )

    test_dataset = prepare_dataset(
        test_records
    )

    print("\n===== LABELS =====")
    print(label2id)

    print("\n===== LOADING PHOBERT =====")

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id,
    )

    # Move model to GPU.
    model = model.to(device)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # TRAINING ARGUMENTS
    # ========================================================

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),

        num_train_epochs=EPOCHS,

        learning_rate=LEARNING_RATE,

        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,

        gradient_accumulation_steps=(
            GRADIENT_ACCUMULATION_STEPS
        ),

        eval_strategy="epoch",
        save_strategy="epoch",

        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,

        weight_decay=0.01,

        logging_strategy="steps",
        logging_steps=10,

        save_total_limit=2,
        report_to="none",

        seed=SEED,

        # GPU acceleration
        use_cpu=False,
        fp16=True,
        dataloader_pin_memory=True,

        # Optimizer
        optim="adamw_torch",
    )

    # ========================================================
    # DATA COLLATOR
    # ========================================================

    collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer
    )

    # ========================================================
    # TRAINER
    # ========================================================

    trainer = Trainer(
        model=model,
        args=args,

        train_dataset=train_dataset,
        eval_dataset=val_dataset,

        data_collator=collator,

        processing_class=tokenizer,

        compute_metrics=compute_metrics,
    )

    print("\n===== TRAINING DEVICE =====")
    print("Trainer device:", trainer.args.device)

    if trainer.args.device.type != "cuda":
        raise RuntimeError(
            "Trainer is not using CUDA!"
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
    # TEST EVALUATION
    # ========================================================

    print("\n===== TEST EVALUATION =====")

    output = trainer.predict(
        test_dataset
    )

    print(output.metrics)

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    print("\n===== CLASSIFICATION REPORT =====")

    pred_tags, gold_tags = decode_predictions(
        output.predictions,
        output.label_ids,
    )

    print(
        classification_report(
            gold_tags,
            pred_tags,
            digits=4,
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
            output.metrics,
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
