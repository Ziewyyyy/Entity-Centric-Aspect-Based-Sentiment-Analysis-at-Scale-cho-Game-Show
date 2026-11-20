
import json
import sys
import gc
from pathlib import Path
from collections import Counter

import torch


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from serving.entity_predictor import EntityPredictor
from serving.opinion_predictor import OpinionPredictor
from serving.relation_predictor import RelationPredictor
from serving.aspect_predictor import AspectPredictor
from serving.sentiment_predictor import SentimentPredictor


DATA_DIR = ROOT / "data" / "labeled"
OUTPUT_DIR = ROOT / "data" / "analysis" / "evaluation"


# ============================================================
# DATA
# ============================================================

def load_jsonl(path):
    records = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return records


def load_test(task):
    return load_jsonl(DATA_DIR / task / "test.jsonl")


# ============================================================
# METRICS
# ============================================================

def safe_div(a, b):
    return a / b if b else 0.0


def set_metrics(gold, predicted):
    gold = set(gold)
    predicted = set(predicted)

    tp = len(gold & predicted)
    fp = len(predicted - gold)
    fn = len(gold - predicted)

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(
        2 * precision * recall,
        precision + recall,
    )

    return {
        "support": len(gold),
        "predicted": len(predicted),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }


def classification_metrics(y_true, y_pred, labels):
    if not y_true:
        return {
            "accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "per_class": {},
            "confusion_matrix": {},
        }

    per_class = {}
    confusion = {
        actual: {predicted: 0 for predicted in labels}
        for actual in labels
    }

    for actual, predicted in zip(y_true, y_pred):
        if actual in confusion and predicted in confusion[actual]:
            confusion[actual][predicted] += 1

    f1_values = []
    precision_values = []
    recall_values = []
    weighted_f1_sum = 0
    total = len(y_true)

    for label in labels:
        tp = sum(
            1
            for a, p in zip(y_true, y_pred)
            if a == label and p == label
        )

        fp = sum(
            1
            for a, p in zip(y_true, y_pred)
            if a != label and p == label
        )

        fn = sum(
            1
            for a, p in zip(y_true, y_pred)
            if a == label and p != label
        )

        support = sum(1 for a in y_true if a == label)

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(
            2 * precision * recall,
            precision + recall,
        )

        per_class[label] = {
            "support": support,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        }

        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
        weighted_f1_sum += f1 * support

    accuracy = safe_div(
        sum(1 for a, p in zip(y_true, y_pred) if a == p),
        total,
    )

    return {
        "samples": total,
        "accuracy": round(accuracy, 6),
        "macro_precision": round(
            sum(precision_values) / len(labels), 6
        ),
        "macro_recall": round(
            sum(recall_values) / len(labels), 6
        ),
        "macro_f1": round(
            sum(f1_values) / len(labels), 6
        ),
        "weighted_f1": round(
            safe_div(weighted_f1_sum, total), 6
        ),
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def release_predictor(predictor):
    del predictor
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# ENTITY EVALUATION
# ============================================================

def evaluate_entity():
    print("\n[1/5] Evaluating Entity...")

    records = load_test("entity")
    predictor = EntityPredictor()

    gold_all = set()
    predicted_all = set()

    for index, record in enumerate(records, 1):
        text = record["text"]
        comment_id = record["comment_id"]

        result = predictor.predict(text)

        for item in record.get("entities", []):
            gold_all.add((
                comment_id,
                item["start"],
                item["end"],
                item["type"],
            ))

        for item in result.get("entities", []):
            predicted_all.add((
                comment_id,
                item["start"],
                item["end"],
                item["type"],
            ))

        if index % 50 == 0:
            print(f"  Processed {index}/{len(records)}")

    metrics = set_metrics(gold_all, predicted_all)

    release_predictor(predictor)

    return {
        "records": len(records),
        "exact_span_metrics": metrics,
    }


# ============================================================
# OPINION EVALUATION
# ============================================================

def evaluate_opinion():
    print("\n[2/5] Evaluating Opinion...")

    records = load_test("opinion")
    predictor = OpinionPredictor()

    gold_all = set()
    predicted_all = set()

    for index, record in enumerate(records, 1):
        text = record["text"]
        comment_id = record["comment_id"]

        result = predictor.predict(text)

        for item in record.get("opinions", []):
            gold_all.add((
                comment_id,
                item["start"],
                item["end"],
            ))

        for item in result.get("opinions", []):
            predicted_all.add((
                comment_id,
                item["start"],
                item["end"],
            ))

        if index % 50 == 0:
            print(f"  Processed {index}/{len(records)}")

    metrics = set_metrics(gold_all, predicted_all)

    release_predictor(predictor)

    return {
        "records": len(records),
        "exact_span_metrics": metrics,
    }


# ============================================================
# RELATION EVALUATION
# ============================================================

def evaluate_relation():
    print("\n[3/5] Evaluating Relation...")

    records = load_test("relation")
    predictor = RelationPredictor()

    y_true = []
    y_pred = []

    for index, record in enumerate(records, 1):
        text = record["text"]
        entities = record.get("entities", [])
        opinions = record.get("opinions", [])

        gold_relations = {
            (item["entity"], item["opinion"])
            for item in record.get("relations", [])
        }

        # Evaluate every possible gold Entity-Opinion pair.
        # This isolates the relation classifier from extraction errors.
        for entity_index, entity in enumerate(entities):
            for opinion_index, opinion in enumerate(opinions):

                result = predictor.predict_pair(
                    text,
                    entity,
                    opinion,
                )

                y_true.append(
                    "HAS_OPINION"
                    if (entity_index, opinion_index) in gold_relations
                    else "NO_RELATION"
                )

                y_pred.append(result["relation"])

        if index % 50 == 0:
            print(f"  Processed {index}/{len(records)}")

    metrics = classification_metrics(
        y_true,
        y_pred,
        ["NO_RELATION", "HAS_OPINION"],
    )

    release_predictor(predictor)

    return {
        "records": len(records),
        "evaluation_method": "Gold entity and opinion spans; all candidate pairs",
        "metrics": metrics,
    }


# ============================================================
# ASPECT EVALUATION
# ============================================================

def evaluate_aspect():
    print("\n[4/5] Evaluating Aspect...")

    records = load_test("aspect")
    predictor = AspectPredictor()

    y_true = []
    y_pred = []
    errors = []

    for index, record in enumerate(records, 1):
        try:
            result = predictor.predict(
                text=record["text"],
                opinion=record["opinion"],
                start=record["start"],
                end=record["end"],
            )

            y_true.append(record["aspect"])
            y_pred.append(result["aspect"])

        except Exception as exc:
            errors.append({
                "comment_id": record.get("comment_id"),
                "error": str(exc),
            })

        if index % 50 == 0:
            print(f"  Processed {index}/{len(records)}")

    metrics = classification_metrics(
        y_true,
        y_pred,
        predictor.id2label.values(),
    )

    release_predictor(predictor)

    return {
        "records": len(records),
        "evaluated": len(y_true),
        "skipped": len(errors),
        "metrics": metrics,
        "errors": errors,
    }


# ============================================================
# SENTIMENT EVALUATION
# ============================================================

def evaluate_sentiment():
    print("\n[5/5] Evaluating Sentiment...")

    records = load_test("sentiment")
    predictor = SentimentPredictor()

    y_true = []
    y_pred = []
    errors = []

    for index, record in enumerate(records, 1):
        try:
            result = predictor.predict(
                text=record["text"],
                opinion=record["opinion"],
                aspect=record["aspect"],
                start=record["start"],
                end=record["end"],
            )

            y_true.append(record["sentiment"])
            y_pred.append(result["sentiment"])

        except Exception as exc:
            errors.append({
                "comment_id": record.get("comment_id"),
                "error": str(exc),
            })

        if index % 50 == 0:
            print(f"  Processed {index}/{len(records)}")

    metrics = classification_metrics(
        y_true,
        y_pred,
        ["NEG", "NEU", "POS"],
    )

    release_predictor(predictor)

    return {
        "records": len(records),
        "evaluated": len(y_true),
        "skipped": len(errors),
        "metrics": metrics,
        "errors": errors,
    }


# ============================================================
# JOINT ASPECT + SENTIMENT EVALUATION
# ============================================================

def evaluate_joint_aspect_sentiment():
    print("\nEvaluating combined Aspect + Sentiment...")

    aspect_records = load_test("aspect")
    sentiment_records = load_test("sentiment")

    def make_key(record):
        return (
            record.get("comment_id"),
            record.get("start"),
            record.get("end"),
            record.get("opinion"),
            record.get("text"),
        )

    sentiment_lookup = {
        make_key(record): record
        for record in sentiment_records
    }

    matched = []

    for aspect_record in aspect_records:
        sentiment_record = sentiment_lookup.get(
            make_key(aspect_record)
        )

        if sentiment_record is not None:
            matched.append((
                aspect_record,
                sentiment_record,
            ))

    print(f"  Matched records: {len(matched)}")

    aspect_predictor = AspectPredictor()
    sentiment_predictor = SentimentPredictor()

    aspect_true = []
    aspect_pred = []

    sentiment_true = []
    sentiment_gold_aspect_pred = []
    sentiment_predicted_aspect_pred = []

    errors = []

    for index, (aspect_record, sentiment_record) in enumerate(
        matched, 1
    ):
        try:
            aspect_result = aspect_predictor.predict(
                text=aspect_record["text"],
                opinion=aspect_record["opinion"],
                start=aspect_record["start"],
                end=aspect_record["end"],
            )

            predicted_aspect = aspect_result["aspect"]

            # Sentiment with the gold aspect:
            # isolates the sentiment model.
            gold_aspect_result = sentiment_predictor.predict(
                text=sentiment_record["text"],
                opinion=sentiment_record["opinion"],
                aspect=sentiment_record["aspect"],
                start=sentiment_record["start"],
                end=sentiment_record["end"],
            )

            # Sentiment with the predicted aspect:
            # measures error propagation from Aspect to Sentiment.
            predicted_aspect_result = sentiment_predictor.predict(
                text=sentiment_record["text"],
                opinion=sentiment_record["opinion"],
                aspect=predicted_aspect,
                start=sentiment_record["start"],
                end=sentiment_record["end"],
            )

            aspect_true.append(aspect_record["aspect"])
            aspect_pred.append(predicted_aspect)

            sentiment_true.append(sentiment_record["sentiment"])
            sentiment_gold_aspect_pred.append(
                gold_aspect_result["sentiment"]
            )
            sentiment_predicted_aspect_pred.append(
                predicted_aspect_result["sentiment"]
            )

        except Exception as exc:
            errors.append({
                "comment_id": aspect_record.get("comment_id"),
                "error": str(exc),
            })

        if index % 50 == 0:
            print(f"  Processed {index}/{len(matched)}")

    aspect_metrics = classification_metrics(
        aspect_true,
        aspect_pred,
        aspect_predictor.id2label.values(),
    )

    sentiment_gold_metrics = classification_metrics(
        sentiment_true,
        sentiment_gold_aspect_pred,
        ["NEG", "NEU", "POS"],
    )

    sentiment_cascade_metrics = classification_metrics(
        sentiment_true,
        sentiment_predicted_aspect_pred,
        ["NEG", "NEU", "POS"],
    )

    release_predictor(aspect_predictor)
    release_predictor(sentiment_predictor)

    return {
        "matched_records": len(matched),
        "evaluated": len(aspect_true),
        "skipped": len(errors),
        "aspect": aspect_metrics,
        "sentiment_with_gold_aspect": sentiment_gold_metrics,
        "sentiment_with_predicted_aspect": sentiment_cascade_metrics,
        "errors": errors,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("ENTITY-CENTRIC ABSA - MODEL EVALUATION")
    print("=" * 65)

    results = {
        "project": "Entity-Centric Aspect-Based Sentiment Analysis",
        "evaluation_split": "test",
        "models": {},
    }

    results["models"]["entity"] = evaluate_entity()
    results["models"]["opinion"] = evaluate_opinion()
    results["models"]["relation"] = evaluate_relation()
    results["models"]["aspect"] = evaluate_aspect()
    results["models"]["sentiment"] = evaluate_sentiment()

    results["combined_aspect_sentiment"] = (
        evaluate_joint_aspect_sentiment()
    )

    output_path = OUTPUT_DIR / "evaluation_results.json"

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 65)
    print("EVALUATION COMPLETED")
    print("=" * 65)

    print("Report:", output_path)

    for name, result in results["models"].items():
        print(f"\n{name.upper()}")

        if "exact_span_metrics" in result:
            metrics = result["exact_span_metrics"]
            print("Precision:", metrics["precision"])
            print("Recall:", metrics["recall"])
            print("F1:", metrics["f1"])

        elif "metrics" in result:
            metrics = result["metrics"]
            print("Accuracy:", metrics["accuracy"])
            print("Macro F1:", metrics["macro_f1"])
            print("Weighted F1:", metrics["weighted_f1"])


if __name__ == "__main__":
    main()
