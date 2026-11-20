"""Relation Classification (HAS_OPINION) - PhoBERT.

Bài toán: với mỗi cặp (entity, opinion) trong cùng một comment, dự đoán
opinion có nói về entity đó hay không.

Input:  comment có đánh dấu <ent> ... </ent> và <op> ... </op>,
        kèm " </s> Entity: <TYPE>"
Output: NO_RELATION | HAS_OPINION

Chạy (từ thư mục gốc project):
    python training/train_relation.py

Dữ liệu: data/labeled/relation/{train,val,test}.jsonl
         (tạo bởi annotation/export_relations.py và split_relation_dataset.py)
"""

import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import relation_common as rc  # noqa: E402
from serving import text_utils as tu  # noqa: E402


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = ROOT / "data" / "labeled" / "relation"
OUTPUT_DIR = ROOT / "models" / "relation_phobert"
ANALYSIS_DIR = ROOT / "data" / "analysis" / "relation"

MODEL_NAME = "vinai/phobert-base-v2"

LABELS = ["NO_RELATION", "HAS_OPINION"]
LABEL2ID = {name: i for i, name in enumerate(LABELS)}
ID2LABEL = {i: name for name, i in LABEL2ID.items()}

MARKERS = list(tu.RELATION_MARKERS)

SEED = 42
MAX_LENGTH = 256
EPOCHS = 8
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 2
LEARNING_RATE = 2e-5


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
        props = torch.cuda.get_device_properties(0)
        print("Device: CUDA GPU")
        print("GPU:", torch.cuda.get_device_name(0))
        print("Total VRAM:", round(props.total_memory / 1024**3, 2), "GB")
    else:
        device = torch.device("cpu")
        print("Device: CPU")
        print("WARNING: CUDA is unavailable.")

    print("=" * 60)
    return device


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# LOAD DATA
# ============================================================

def load_records(path):
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    records = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
                rc.validate_record(record)
            except (json.JSONDecodeError, ValueError, KeyError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error

            records.append(record)

    if not records:
        raise ValueError(f"No records in {path}")

    return records


def make_pairs(records):
    pairs = []
    for record in records:
        pairs.extend(rc.build_pairs(record))
    return pairs


# ============================================================
# BUILD DATASET
# ============================================================

def build_dataset(pairs, tokenizer, split):
    """Mã hóa các cặp. Loại cặp bị cắt mất marker (comment quá dài)."""
    texts = [
        tu.build_relation_input(p["text"], p["entity"], p["opinion"])
        for p in pairs
    ]

    encoded = tokenizer(texts, truncation=True, max_length=MAX_LENGTH)
    marker_ids = [tokenizer.convert_tokens_to_ids(m) for m in MARKERS]

    kept = [
        i
        for i, ids in enumerate(encoded["input_ids"])
        if all(m in ids for m in marker_ids)
    ]

    dropped = len(pairs) - len(kept)
    if dropped:
        print(f"[WARNING] {split}: dropped {dropped} pairs (markers truncated)")

    kept_pairs = [pairs[i] for i in kept]

    dataset = Dataset.from_dict({
        "input_ids": [encoded["input_ids"][i] for i in kept],
        "attention_mask": [encoded["attention_mask"][i] for i in kept],
        "labels": [kept_pairs[j]["label"] for j in range(len(kept))],
    })

    return dataset, kept_pairs


# ============================================================
# METRICS
# ============================================================

def metrics_fn(prediction):
    logits = prediction.predictions
    if isinstance(logits, tuple):
        logits = logits[0]

    y_pred = np.argmax(logits, axis=-1)
    y_true = prediction.label_ids

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=[0, 1], average="macro", zero_division=0),
        "pos_precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "pos_recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "pos_f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
    }


def positive_probabilities(logits):
    if isinstance(logits, tuple):
        logits = logits[0]

    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(logits)
    return (exp[:, 1] / exp.sum(axis=-1)).tolist()


# ============================================================
# WEIGHTED TRAINER
# ============================================================

class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        weights = self.class_weights.to(device=logits.device, dtype=torch.float32)
        loss = F.cross_entropy(logits.float(), labels, weight=weights)

        return (loss, outputs) if return_outputs else loss


def make_weights(pairs):
    counts = Counter(p["label"] for p in pairs)

    print("\nTRAIN CLASS DISTRIBUTION")
    for i, name in enumerate(LABELS):
        print(f"{name:12s}: {counts.get(i, 0)}")

    if counts.get(0, 0) == 0 or counts.get(1, 0) == 0:
        raise ValueError(f"Train set needs both classes: {dict(counts)}")

    total = sum(counts.values())
    weights = [
        (total / (len(LABELS) * counts[i])) ** 0.5
        for i in range(len(LABELS))
    ]

    print("\nCLASS WEIGHTS")
    for name, weight in zip(LABELS, weights):
        print(f"{name:12s}: {weight:.3f}")

    return torch.tensor(weights, dtype=torch.float32)


# ============================================================
# EVALUATION AT OPINION LEVEL
# ============================================================

def evaluate_groups(name, records, all_pairs, kept_pairs, probs, threshold):
    """So model với baseline 'entity gần nhất' trên từng opinion."""
    keys = rc.complete_keys(all_pairs, kept_pairs)
    hard_keys = rc.multi_entity_keys(all_pairs) & keys

    gold = rc.gold_groups(all_pairs)
    model_pred = rc.predicted_groups(kept_pairs, probs, threshold)
    baseline_pred = rc.nearest_groups(records)

    result = {}

    for subset_name, subset in (("all", keys), ("multi_entity", hard_keys)):
        model_score = rc.score_groups(gold, model_pred, subset)
        baseline_score = rc.score_groups(gold, baseline_pred, subset)
        result[subset_name] = {"model": model_score, "baseline": baseline_score}

        print(f"\n[{name}] subset={subset_name} ({model_score['opinions']} opinions)")
        print(f"{'':12s}{'exact':>8s}{'prec':>8s}{'recall':>8s}{'f1':>8s}")
        for label, score in (("model", model_score), ("nearest", baseline_score)):
            print(
                f"{label:12s}"
                f"{score['exact_match']:8.3f}"
                f"{score['link_precision']:8.3f}"
                f"{score['link_recall']:8.3f}"
                f"{score['link_f1']:8.3f}"
            )

    return result, model_pred, gold


def save_error_analysis(records, kept_pairs, model_pred, gold, probs, threshold):
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    by_id = {str(r["comment_id"]): r for r in records}
    prob_by_pair = {
        (p["comment_id"], p["opinion_index"], p["entity_index"]): prob
        for p, prob in zip(kept_pairs, probs)
    }

    path = ANALYSIS_DIR / "test_errors.csv"

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "comment_id", "text", "opinion", "gold_entities", "predicted_entities", "probabilities",
        ])

        for key, predicted in model_pred.items():
            if predicted == gold.get(key):
                continue

            comment_id, opinion_index = key
            record = by_id[comment_id]
            entities = record["entities"]

            def names(indexes):
                return " | ".join(entities[i]["text"] for i in sorted(indexes))

            probabilities = " | ".join(
                f"{entities[i]['text']}={prob_by_pair[(comment_id, opinion_index, i)]:.2f}"
                for i in range(len(entities))
                if (comment_id, opinion_index, i) in prob_by_pair
            )

            writer.writerow([
                comment_id,
                record["text"],
                record["opinions"][opinion_index]["text"],
                names(gold.get(key, set())),
                names(predicted),
                probabilities,
            ])

    print("Error analysis saved to:", path)


# ============================================================
# MAIN
# ============================================================

def main():
    device = setup_device()
    seed_everything(SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------
    print("\nLoading datasets...")

    train_records = load_records(DATA_DIR / "train.jsonl")
    val_records = load_records(DATA_DIR / "val.jsonl")
    test_records = load_records(DATA_DIR / "test.jsonl")

    id_sets = [
        {str(r["comment_id"]) for r in records}
        for records in (train_records, val_records, test_records)
    ]

    if any(id_sets[i] & id_sets[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Comment ID leakage across splits")

    train_pairs = make_pairs(train_records)
    val_pairs = make_pairs(val_records)
    test_pairs = make_pairs(test_records)

    print(
        f"Comments: train={len(train_records)}, "
        f"val={len(val_records)}, test={len(test_records)}"
    )
    print(
        f"Pairs:    train={len(train_pairs)}, "
        f"val={len(val_pairs)}, test={len(test_pairs)}"
    )

    for name, pairs in (("train", train_pairs), ("val", val_pairs), ("test", test_pairs)):
        positives = sum(p["label"] for p in pairs)
        print(f"  {name}: positive={positives}, negative={len(pairs) - positives}")

    # --------------------------------------------------------
    # TOKENIZER
    # --------------------------------------------------------
    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    tokenizer.add_special_tokens({"additional_special_tokens": MARKERS})

    train_data, train_kept = build_dataset(train_pairs, tokenizer, "train")
    val_data, val_kept = build_dataset(val_pairs, tokenizer, "val")
    test_data, test_kept = build_dataset(test_pairs, tokenizer, "test")

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
    model.resize_token_embeddings(len(tokenizer))

    weights = make_weights(train_kept)

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------
    use_cuda = device.type == "cuda"

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        logging_strategy="epoch",
        report_to="none",
        seed=SEED,
        fp16=use_cuda,
        dataloader_pin_memory=use_cuda,
        optim="adamw_torch",
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=val_data,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=metrics_fn,
        class_weights=weights,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("\n" + "=" * 60)
    print("START TRAINING")
    print("=" * 60)

    trainer.train()

    # --------------------------------------------------------
    # VALIDATION: chọn ngưỡng quyết định
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("VALIDATION - BEST CHECKPOINT")
    print("=" * 60)

    val_output = trainer.predict(val_data)
    print(val_output.metrics)

    val_probs = positive_probabilities(val_output.predictions)
    val_keys = rc.complete_keys(val_pairs, val_kept)

    threshold, table = rc.find_best_threshold(val_kept, val_probs, val_keys)

    print("\nTHRESHOLD SEARCH (validation, link F1)")
    for row in table:
        marker = "  <== chosen" if row["threshold"] == threshold else ""
        print(
            f"  threshold={row['threshold']:.2f}  "
            f"exact={row['exact_match']:.3f}  f1={row['link_f1']:.3f}{marker}"
        )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"TEST - FINAL (threshold={threshold})")
    print("=" * 60)

    test_output = trainer.predict(test_data)
    test_probs = positive_probabilities(test_output.predictions)

    y_true = test_output.label_ids
    y_pred = [int(p >= threshold) for p in test_probs]

    print("\nPAIR-LEVEL REPORT")
    print(
        classification_report(
            y_true, y_pred, labels=[0, 1], target_names=LABELS, zero_division=0
        )
    )

    print("OPINION-LEVEL COMPARISON (model vs nearest-entity baseline)")
    group_result, model_pred, gold = evaluate_groups(
        "test", test_records, test_pairs, test_kept, test_probs, threshold
    )

    save_error_analysis(test_records, test_kept, model_pred, gold, test_probs, threshold)

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    (OUTPUT_DIR / "relation_config.json").write_text(
        json.dumps(
            {
                "labels": LABELS,
                "threshold": threshold,
                "markers": MARKERS,
                "max_length": MAX_LENGTH,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    (OUTPUT_DIR / "test_metrics.json").write_text(
        json.dumps(
            {
                "pair_level": test_output.metrics,
                "opinion_level": group_result,
                "threshold_search": table,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)
    print("Model saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
