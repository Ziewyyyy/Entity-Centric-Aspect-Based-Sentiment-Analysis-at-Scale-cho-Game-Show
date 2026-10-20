
"""Aspect Classification V2 - PhoBERT GPU Optimized.

Features:
- PhoBERT-base-v2
- Vietnamese word segmentation
- Opinion span markers
- Weighted cross-entropy loss
- GPU CUDA / CPU fallback
- FP16 mixed precision
- Gradient accumulation
- Gradient checkpointing
- Early stopping
- Validation and test evaluation
- Confusion matrix and prediction exports

Run:
    python training/train_aspect_v2.py
"""

import csv
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from underthesea import word_tokenize


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "labeled" / "aspect"
OUTPUT_DIR = ROOT / "models" / "aspect_phobert_v2"
ANALYSIS_DIR = ROOT / "data" / "analysis" / "aspect_v2"

MODEL_NAME = "vinai/phobert-base-v2"

LABELS = [
    "VOCAL",
    "DANCE",
    "PERFORMANCE",
    "VISUAL",
    "PERSONALITY",
    "INTERACTION",
    "CONTENT",
    "PRODUCTION",
    "EDITING",
    "GENERAL",
    "POPULARITY",
    "AWARD_RESULT",
]

LABEL2ID = {name: i for i, name in enumerate(LABELS)}
ID2LABEL = {i: name for name, i in LABEL2ID.items()}

SEED = 42

# RTX 3050 4GB settings
MAX_LENGTH = 256
EPOCHS = 8
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 2
LEARNING_RATE = 2e-5

MARK_START = "<opinion>"
MARK_END = "</opinion>"


# ============================================================
# DEVICE
# ============================================================

def setup_device():
    print("\n" + "=" * 60)
    print("DEVICE INFORMATION")
    print("=" * 60)

    print("PyTorch:", torch.__version__)

    if torch.cuda.is_available():
        device = torch.device("cuda")

        print("Device: CUDA GPU")
        print("GPU:", torch.cuda.get_device_name(0))

        props = torch.cuda.get_device_properties(0)

        print(
            "Total VRAM:",
            round(props.total_memory / 1024**3, 2),
            "GB",
        )

        free, total = torch.cuda.mem_get_info()

        print(
            "Free VRAM:",
            round(free / 1024**3, 2),
            "GB",
        )

        if free < 2 * 1024**3:
            print(
                "WARNING: Low free VRAM. "
                "Close games and other GPU applications."
            )

    else:
        device = torch.device("cpu")
        print("Device: CPU")
        print("WARNING: CUDA is unavailable.")

    print("=" * 60)

    return device


# ============================================================
# RANDOM SEED
# ============================================================

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# READ DATA
# ============================================================

def read_records(path):
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    records = []

    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue

            record = json.loads(line)

            if record.get("aspect") not in LABEL2ID:
                raise ValueError(
                    f"{path}:{lineno}: "
                    f"Unknown aspect: {record.get('aspect')}"
                )

            if not isinstance(record.get("text"), str):
                raise ValueError(
                    f"{path}:{lineno}: Missing text"
                )

            if not isinstance(record.get("opinion"), str):
                raise ValueError(
                    f"{path}:{lineno}: Missing opinion"
                )

            records.append(record)

    if not records:
        raise ValueError(f"No records in {path}")

    return records


# ============================================================
# WORD SEGMENTATION
# ============================================================

def segment(text):
    if not text.strip():
        return ""

    return word_tokenize(text, format="text")


# ============================================================
# PREPARE INPUT
# ============================================================

def prepare_input(record):
    text = record["text"]
    opinion = record["opinion"]

    start = record.get("start")
    end = record.get("end")

    if not isinstance(start, int) or not isinstance(end, int):
        raise ValueError(
            f"Missing start/end offsets: "
            f"{record.get('comment_id')}"
        )

    if not (0 <= start < end <= len(text)):
        raise ValueError(
            f"Invalid offsets: {record.get('comment_id')}"
        )

    if text[start:end] != opinion:
        raise ValueError(
            f"Opinion mismatch: {record.get('comment_id')}"
        )

    before = segment(text[:start].strip())
    target = segment(text[start:end].strip())
    after = segment(text[end:].strip())

    context = " ".join(
        x for x in (
            before,
            MARK_START,
            target,
            MARK_END,
            after,
        )
        if x
    )

    return context, segment(opinion)


# ============================================================
# BUILD DATASET
# ============================================================

def build_dataset(records, tokenizer):
    inputs = [prepare_input(r) for r in records]

    texts = [
        f"{context} </s> {opinion}"
        for context, opinion in inputs
    ]

    encoded = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_LENGTH,
    )

    encoded["labels"] = [
        LABEL2ID[r["aspect"]]
        for r in records
    ]

    return Dataset.from_dict(encoded)


# ============================================================
# METRICS
# ============================================================

def metrics_fn(pred):
    predictions = pred.predictions

    if isinstance(predictions, tuple):
        predictions = predictions[0]

    y_pred = np.argmax(predictions, axis=-1)
    y_true = pred.label_ids

    return {
        "accuracy": accuracy_score(y_true, y_pred),

        "macro_f1": f1_score(
            y_true,
            y_pred,
            labels=list(range(len(LABELS))),
            average="macro",
            zero_division=0,
        ),

        "weighted_f1": f1_score(
            y_true,
            y_pred,
            labels=list(range(len(LABELS))),
            average="weighted",
            zero_division=0,
        ),
    }


# ============================================================
# WEIGHTED TRAINER
# ============================================================

class WeightedTrainer(Trainer):

    def __init__(
        self,
        *args,
        class_weights=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        labels = inputs.pop("labels")

        outputs = model(**inputs)
        logits = outputs.logits

        weights = self.class_weights.to(
            device=logits.device,
            dtype=torch.float32,
        )

        loss = F.cross_entropy(
            logits.float(),
            labels,
            weight=weights,
        )

        return (loss, outputs) if return_outputs else loss


# ============================================================
# CLASS WEIGHTS
# ============================================================

def make_weights(train_records):
    counts = Counter(
        r["aspect"] for r in train_records
    )

    print("\nTRAIN CLASS DISTRIBUTION")

    for label in LABELS:
        print(f"{label:15s}: {counts.get(label, 0)}")

    values = np.array(
        [counts.get(label, 0) for label in LABELS],
        dtype=np.float64,
    )

    if np.any(values == 0):
        missing = [
            label
            for label, count in zip(LABELS, values)
            if count == 0
        ]

        raise ValueError(
            f"Classes absent from TRAIN: {missing}"
        )

    weights = np.sqrt(
        values.sum() / (len(LABELS) * values)
    )

    weights = np.minimum(weights, 4.0)
    weights = weights / weights.mean()

    print("\nCLASS WEIGHTS")

    for label, weight in zip(LABELS, weights):
        print(f"{label:15s}: {weight:.3f}")

    return torch.tensor(
        weights,
        dtype=torch.float32,
    )


# ============================================================
# SAVE ANALYSIS
# ============================================================

def save_analysis(records, predictions):
    ANALYSIS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    true_ids = [
        LABEL2ID[r["aspect"]]
        for r in records
    ]

    matrix = confusion_matrix(
        true_ids,
        predictions,
        labels=list(range(len(LABELS))),
    )

    matrix_path = ANALYSIS_DIR / "confusion_matrix.csv"

    with matrix_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            ["gold\\pred"] + LABELS
        )

        for label, row in zip(LABELS, matrix):
            writer.writerow(
                [label] + row.tolist()
            )

    prediction_path = (
        ANALYSIS_DIR / "test_predictions.csv"
    )

    with prediction_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        fields = [
            "comment_id",
            "text",
            "opinion",
            "start",
            "end",
            "gold_aspect",
            "predicted_aspect",
            "correct",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        for record, pred in zip(records, predictions):
            predicted_aspect = ID2LABEL[int(pred)]

            writer.writerow({
                "comment_id": record.get("comment_id", ""),
                "text": record["text"],
                "opinion": record["opinion"],
                "start": record.get("start", ""),
                "end": record.get("end", ""),
                "gold_aspect": record["aspect"],
                "predicted_aspect": predicted_aspect,
                "correct": (
                    record["aspect"] == predicted_aspect
                ),
            })

    print("\nAnalysis saved to:", ANALYSIS_DIR)


# ============================================================
# MAIN TRAINING
# ============================================================

def main():

    device = setup_device()

    seed_everything(SEED)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    print("\nLoading datasets...")

    train_records = read_records(
        DATA_DIR / "train.jsonl"
    )

    val_records = read_records(
        DATA_DIR / "val.jsonl"
    )

    test_records = read_records(
        DATA_DIR / "test.jsonl"
    )

    print(
        f"Samples: "
        f"train={len(train_records)}, "
        f"val={len(val_records)}, "
        f"test={len(test_records)}"
    )

    # --------------------------------------------------------
    # CHECK DATA LEAKAGE
    # --------------------------------------------------------

    groups = [
        {
            str(r.get("comment_id"))
            for r in records
        }
        for records in (
            train_records,
            val_records,
            test_records,
        )
    ]

    if any(
        groups[i] & groups[j]
        for i, j in (
            (0, 1),
            (0, 2),
            (1, 2),
        )
    ):
        raise ValueError(
            "Comment ID leakage across splits"
        )

    # --------------------------------------------------------
    # TOKENIZER
    # --------------------------------------------------------

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=False,
    )

    tokenizer.add_special_tokens({
        "additional_special_tokens": [
            MARK_START,
            MARK_END,
        ]
    })

    print("\nTokenizing datasets...")

    train_data = build_dataset(
        train_records,
        tokenizer,
    )

    val_data = build_dataset(
        val_records,
        tokenizer,
    )

    test_data = build_dataset(
        test_records,
        tokenizer,
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print("\nLoading PhoBERT model...")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        label2id=LABEL2ID,
        id2label=ID2LABEL,
    )

    model.resize_token_embeddings(
        len(tokenizer)
    )

    # Disable cache for gradient checkpointing
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    # --------------------------------------------------------
    # CLASS WEIGHTS
    # --------------------------------------------------------

    weights = make_weights(
        train_records
    )

    # --------------------------------------------------------
    # TRAINING CONFIGURATION
    # --------------------------------------------------------

    use_cuda = device.type == "cuda"

    print("\nTRAINING CONFIGURATION")
    print("Model:", MODEL_NAME)
    print("Device:", device)
    print("Epochs:", EPOCHS)
    print("Batch size:", BATCH_SIZE)
    print(
        "Gradient accumulation:",
        GRADIENT_ACCUMULATION,
    )
    print(
        "Effective batch size:",
        BATCH_SIZE * GRADIENT_ACCUMULATION,
    )
    print("Max length:", MAX_LENGTH)
    print("FP16:", use_cuda)

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),

        num_train_epochs=EPOCHS,

        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,

        gradient_accumulation_steps=GRADIENT_ACCUMULATION,

        gradient_checkpointing=use_cuda,

        learning_rate=LEARNING_RATE,

        eval_strategy="epoch",
        save_strategy="epoch",

        load_best_model_at_end=True,

        metric_for_best_model="macro_f1",
        greater_is_better=True,

        save_total_limit=2,

        report_to="none",

        seed=SEED,

        weight_decay=0.01,

        logging_strategy="epoch",

        fp16=use_cuda,

        dataloader_pin_memory=use_cuda,

        optim="adamw_torch",

        # Weighted loss uses mean reduction independently
        # of gradient accumulation batch size.
        average_tokens_across_devices=False,
    )

    # --------------------------------------------------------
    # TRAINER
    # --------------------------------------------------------

    trainer = WeightedTrainer(
        model=model,

        args=args,

        train_dataset=train_data,
        eval_dataset=val_data,

        processing_class=tokenizer,

        data_collator=DataCollatorWithPadding(
            tokenizer
        ),

        compute_metrics=metrics_fn,

        class_weights=weights,

        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=2
            )
        ],
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("START TRAINING")
    print("=" * 60)

    trainer.train()

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("VALIDATION - BEST CHECKPOINT")
    print("=" * 60)

    validation_metrics = trainer.evaluate(
        val_data
    )

    print(validation_metrics)

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("TEST - FINAL")
    print("=" * 60)

    result = trainer.predict(
        test_data
    )

    logits = result.predictions

    if isinstance(logits, tuple):
        logits = logits[0]

    preds = np.argmax(
        logits,
        axis=-1,
    )

    report = classification_report(
        result.label_ids,
        preds,
        labels=list(range(len(LABELS))),
        target_names=LABELS,
        zero_division=0,
    )

    print(report)

    # --------------------------------------------------------
    # SAVE ANALYSIS
    # --------------------------------------------------------

    save_analysis(
        test_records,
        preds,
    )

    # Save classification report
    ANALYSIS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with (
        ANALYSIS_DIR / "classification_report.txt"
    ).open("w", encoding="utf-8") as f:
        f.write(report)

    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    trainer.save_model(
        str(OUTPUT_DIR)
    )

    tokenizer.save_pretrained(
        str(OUTPUT_DIR)
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)

    print("Model saved to:", OUTPUT_DIR)
    print("Analysis saved to:", ANALYSIS_DIR)


if __name__ == "__main__":
    main()
