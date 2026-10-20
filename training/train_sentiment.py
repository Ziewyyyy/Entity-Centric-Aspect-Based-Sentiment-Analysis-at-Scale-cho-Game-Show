
import json
import inspect
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "labeled" / "sentiment"
OUTPUT_DIR = ROOT / "models" / "sentiment_phobert"
ANALYSIS_DIR = ROOT / "data" / "analysis" / "sentiment"

MODEL_NAME = "vinai/phobert-base-v2"

LABELS = ["NEG", "NEU", "POS"]

LABEL2ID = {
    label: index
    for index, label in enumerate(LABELS)
}

ID2LABEL = {
    index: label
    for label, index in LABEL2ID.items()
}

MAX_LENGTH = 192
EPOCHS = 5
BATCH_SIZE = 4
EVAL_BATCH_SIZE = 8
GRAD_ACCUMULATION = 2
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
SEED = 42

# "sqrt" = làm mềm trọng số lớp hiếm
# "none" = không dùng weighted loss
# "inverse" = nghịch đảo tần suất
WEIGHT_MODE = "sqrt"

OP_START = "<op>"
OP_END = "</op>"

ASPECTS = {
    "GENERAL",
    "VOCAL",
    "CONTENT",
    "VISUAL",
    "POPULARITY",
    "PRODUCTION",
    "PERFORMANCE",
    "DANCE",
    "INTERACTION",
    "PERSONALITY",
    "AWARD_RESULT",
    "EDITING",
}


def load_jsonl(path):
    if not path.exists():
        raise FileNotFoundError(path)

    with open(path, encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )


def save_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(row, ensure_ascii=False) + "\n"
            )


def validate_rows(rows, split):
    for index, row in enumerate(rows, start=1):
        for key in (
            "comment_id",
            "text",
            "opinion",
            "start",
            "end",
            "aspect",
            "sentiment",
        ):
            if key not in row:
                raise ValueError(
                    f"{split} row {index}: missing {key}"
                )

        if row["sentiment"] not in LABEL2ID:
            raise ValueError(
                f"{split} row {index}: "
                f"invalid sentiment {row['sentiment']}"
            )

        if row["aspect"] not in ASPECTS:
            raise ValueError(
                f"{split} row {index}: "
                f"invalid aspect {row['aspect']}"
            )

        text = row["text"]
        start = row["start"]
        end = row["end"]

        if not (
            isinstance(start, int)
            and isinstance(end, int)
            and 0 <= start < end <= len(text)
        ):
            raise ValueError(
                f"{split} row {index}: invalid offsets"
            )

        if text[start:end] != row["opinion"]:
            raise ValueError(
                f"{split} row {index}: "
                "opinion span does not match text"
            )


def build_input(row):
    """
    Mark the target Opinion Span in the comment.

    Example:
    Isaac hát hay nhưng nhảy chưa tốt.

    Target: hay, Aspect: VOCAL

    Input:
    Isaac hát <op> hay </op> nhưng nhảy chưa tốt.
    </s> Aspect: VOCAL
    """

    text = row["text"]
    start = row["start"]
    end = row["end"]

    marked_text = (
        text[:start]
        + f" {OP_START} "
        + text[start:end]
        + f" {OP_END} "
        + text[end:]
    )

    return (
        marked_text
        + " </s> Aspect: "
        + row["aspect"]
    )


def prepare_dataset(rows, tokenizer):
    examples = [
        {
            "input_text": build_input(row),
            "labels": LABEL2ID[row["sentiment"]],
        }
        for row in rows
    ]

    dataset = Dataset.from_list(examples)

    def tokenize(batch):
        return tokenizer(
            batch["input_text"],
            truncation=True,
            max_length=MAX_LENGTH,
        )

    dataset = dataset.map(
        tokenize,
        batched=True,
        remove_columns=["input_text"],
    )

    return dataset


def calculate_class_weights(train_rows):
    counts = Counter(
        row["sentiment"]
        for row in train_rows
    )

    total = len(train_rows)
    num_classes = len(LABELS)

    weights = []

    print("\n===== CLASS WEIGHTS =====")

    for label in LABELS:
        count = counts[label]

        if count == 0:
            raise ValueError(
                f"Train set has no examples for {label}"
            )

        inverse_weight = total / (
            num_classes * count
        )

        if WEIGHT_MODE == "sqrt":
            weight = inverse_weight ** 0.5
        elif WEIGHT_MODE == "inverse":
            weight = inverse_weight
        elif WEIGHT_MODE == "none":
            weight = 1.0
        else:
            raise ValueError(
                f"Unknown WEIGHT_MODE: {WEIGHT_MODE}"
            )

        weights.append(weight)

        print(
            f"{label}: count={count}, "
            f"weight={weight:.4f}"
        )

    return torch.tensor(
        weights,
        dtype=torch.float32,
    )


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        labels = inputs["labels"]

        model_inputs = {
            key: value
            for key, value in inputs.items()
            if key != "labels"
        }

        outputs = model(**model_inputs)

        logits = outputs.logits

        weights = self.class_weights.to(
            device=logits.device,
            dtype=torch.float32,
        )

        loss_fn = nn.CrossEntropyLoss(
            weight=weights
        )

        loss = loss_fn(
            logits.float(),
            labels,
        )

        return (
            (loss, outputs)
            if return_outputs
            else loss
        )


def compute_metrics(eval_prediction):
    logits, labels = eval_prediction

    predictions = np.argmax(
        logits,
        axis=-1,
    )

    precision, recall, macro_f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            labels=list(range(len(LABELS))),
            average="macro",
            zero_division=0,
        )
    )

    weighted_f1 = f1_score(
        labels,
        predictions,
        labels=list(range(len(LABELS))),
        average="weighted",
        zero_division=0,
    )

    accuracy = accuracy_score(
        labels,
        predictions,
    )

    return {
        "accuracy": float(accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
    }


def make_training_args():
    """
    Compatible with Transformers versions using either
    eval_strategy or evaluation_strategy.
    """

    parameters = inspect.signature(
        TrainingArguments.__init__
    ).parameters

    kwargs = {
        "output_dir": str(OUTPUT_DIR / "checkpoints"),
        "num_train_epochs": EPOCHS,
        "per_device_train_batch_size": BATCH_SIZE,
        "per_device_eval_batch_size": EVAL_BATCH_SIZE,
        "gradient_accumulation_steps": GRAD_ACCUMULATION,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "save_strategy": "epoch",
        "logging_strategy": "steps",
        "logging_steps": 20,
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "save_total_limit": 2,
        "fp16": torch.cuda.is_available(),
        "dataloader_num_workers": 0,
        "report_to": "none",
        "seed": SEED,
        "data_seed": SEED,
    }

    if "eval_strategy" in parameters:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"

    if "optim" in parameters:
        kwargs["optim"] = "adamw_torch"

    return TrainingArguments(**kwargs)


def evaluate_test(trainer, test_dataset, test_rows):
    print("\n===== FINAL TEST EVALUATION =====")

    output = trainer.predict(test_dataset)

    logits = output.predictions

    predicted_ids = np.argmax(
        logits,
        axis=-1,
    )

    gold_ids = np.array([
        LABEL2ID[row["sentiment"]]
        for row in test_rows
    ])

    probabilities = torch.softmax(
        torch.tensor(logits, dtype=torch.float32),
        dim=-1,
    ).numpy()

    report_text = classification_report(
        gold_ids,
        predicted_ids,
        labels=list(range(len(LABELS))),
        target_names=LABELS,
        zero_division=0,
    )

    report_dict = classification_report(
        gold_ids,
        predicted_ids,
        labels=list(range(len(LABELS))),
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(
        gold_ids,
        predicted_ids,
        labels=list(range(len(LABELS))),
    )

    print(report_text)

    print("\nConfusion Matrix")
    print("Label order:", LABELS)
    print(cm)

    save_json(
        ANALYSIS_DIR / "classification_report.json",
        report_dict,
    )

    save_json(
        ANALYSIS_DIR / "confusion_matrix.json",
        {
            "labels": LABELS,
            "matrix": cm.tolist(),
        },
    )

    predictions = []
    errors = []

    for row, gold, pred, probs in zip(
        test_rows,
        gold_ids,
        predicted_ids,
        probabilities,
    ):
        item = {
            **row,
            "gold_sentiment": ID2LABEL[int(gold)],
            "predicted_sentiment": ID2LABEL[int(pred)],
            "confidence": float(np.max(probs)),
            "probabilities": {
                LABELS[i]: float(probs[i])
                for i in range(len(LABELS))
            },
            "correct": bool(gold == pred),
        }

        predictions.append(item)

        if gold != pred:
            errors.append(item)

    save_jsonl(
        ANALYSIS_DIR / "test_predictions.jsonl",
        predictions,
    )

    save_jsonl(
        ANALYSIS_DIR / "test_errors.jsonl",
        errors,
    )

    print("\nTest samples:", len(test_rows))
    print("Prediction errors:", len(errors))

    return report_dict


def main():
    set_seed(SEED)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ANALYSIS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 60)
    print("PHOBERT SENTIMENT CLASSIFICATION")
    print("=" * 60)

    print("\n===== GPU CONFIGURATION =====")

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )
        print(
            "VRAM:",
            round(
                torch.cuda.get_device_properties(0).total_memory
                / (1024 ** 3),
                2,
            ),
            "GB",
        )
        print("Mixed Precision: FP16")
    else:
        print("Device: CPU")
        print("Mixed Precision: Disabled")

    print("\n===== LOADING DATASETS =====")

    train_rows = load_jsonl(
        DATA_DIR / "train.jsonl"
    )

    val_rows = load_jsonl(
        DATA_DIR / "val.jsonl"
    )

    test_rows = load_jsonl(
        DATA_DIR / "test.jsonl"
    )

    for split, rows in (
        ("TRAIN", train_rows),
        ("VAL", val_rows),
        ("TEST", test_rows),
    ):
        validate_rows(rows, split)

        print(
            f"{split}: {len(rows)} spans, "
            f"{dict(Counter(r['sentiment'] for r in rows))}"
        )

    print("\n===== LOADING TOKENIZER =====")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=False,
    )

    tokenizer.add_special_tokens({
        "additional_special_tokens": [
            OP_START,
            OP_END,
        ]
    })

    print("Tokenizer size:", len(tokenizer))

    print("\n===== PREPARING DATASETS =====")

    train_dataset = prepare_dataset(
        train_rows,
        tokenizer,
    )

    val_dataset = prepare_dataset(
        val_rows,
        tokenizer,
    )

    test_dataset = prepare_dataset(
        test_rows,
        tokenizer,
    )

    print("\n===== LOADING MODEL =====")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    model.resize_token_embeddings(
        len(tokenizer)
    )

    class_weights = calculate_class_weights(
        train_rows
    )

    args = make_training_args()

    collator = DataCollatorWithPadding(
        tokenizer=tokenizer
    )

    trainer_kwargs = {
        "model": model,
        "args": args,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "data_collator": collator,
        "compute_metrics": compute_metrics,
        "callbacks": [
            EarlyStoppingCallback(
                early_stopping_patience=2
            )
        ],
        "class_weights": class_weights,
    }

    trainer_parameters = inspect.signature(
        Trainer.__init__
    ).parameters

    if "processing_class" in trainer_parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = WeightedTrainer(
        **trainer_kwargs
    )

    print("\n===== TRAINING =====")

    trainer.train()

    print("\n===== SAVING BEST MODEL =====")

    trainer.save_model(
        str(OUTPUT_DIR)
    )

    tokenizer.save_pretrained(
        str(OUTPUT_DIR)
    )

    save_json(
        ANALYSIS_DIR / "training_config.json",
        {
            "model_name": MODEL_NAME,
            "labels": LABELS,
            "max_length": MAX_LENGTH,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "gradient_accumulation": GRAD_ACCUMULATION,
            "learning_rate": LEARNING_RATE,
            "weight_mode": WEIGHT_MODE,
            "class_weights": class_weights.tolist(),
            "train_samples": len(train_rows),
            "val_samples": len(val_rows),
            "test_samples": len(test_rows),
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "best_metric": trainer.state.best_metric,
        },
    )

    print("Best checkpoint:", trainer.state.best_model_checkpoint)
    print("Best validation metric:", trainer.state.best_metric)

    evaluate_test(
        trainer,
        test_dataset,
        test_rows,
    )

    print("\n" + "=" * 60)
    print("SENTIMENT TRAINING COMPLETED")
    print("=" * 60)

    print("Model:", OUTPUT_DIR)
    print("Analysis:", ANALYSIS_DIR)


if __name__ == "__main__":
    main()
