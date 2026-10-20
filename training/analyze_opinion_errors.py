
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from predict_opinion import OpinionPredictor


DEFAULT_TEST_PATH = Path("data/labeled/opinion/test.jsonl")
DEFAULT_MODEL_PATH = Path("D:/AI_Models/opinion_phobert")
DEFAULT_OUTPUT_DIR = Path("data/analysis/opinion")


def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_number}: {exc}"
                ) from exc

    return records


def normalize_span(span, text):
    start = int(span["start"])
    end = int(span["end"])

    if not (0 <= start < end <= len(text)):
        raise ValueError(
            f"Invalid span [{start}, {end}] "
            f"for text of length {len(text)}"
        )

    return {
        "text": text[start:end],
        "start": start,
        "end": end,
    }


def get_gold_opinions(record):
    text = record["text"]

    # Dataset được xuất từ export_opinion_spans.py
    # Giả định trường nhãn là "opinions".
    raw_opinions = record.get("opinions")

    if raw_opinions is None:
        raise KeyError(
            "Missing 'opinions' field. "
            "Check your test.jsonl schema."
        )

    return [
        normalize_span(span, text)
        for span in raw_opinions
    ]


def span_key(span):
    return (span["start"], span["end"])


def spans_overlap(a, b):
    return (
        a["start"] < b["end"]
        and b["start"] < a["end"]
    )


def span_to_string(span):
    return (
        f"{span['text']} "
        f"[{span['start']}:{span['end']}]"
    )


def analyze_comment(gold, predicted):
    gold_map = {span_key(s): s for s in gold}
    pred_map = {span_key(s): s for s in predicted}

    gold_keys = set(gold_map)
    pred_keys = set(pred_map)

    tp_keys = gold_keys & pred_keys
    fp_keys = pred_keys - gold_keys
    fn_keys = gold_keys - pred_keys

    tp = [gold_map[k] for k in sorted(tp_keys)]
    fp = [pred_map[k] for k in sorted(fp_keys)]
    fn = [gold_map[k] for k in sorted(fn_keys)]

    # Một span có thể bị tính là FP và FN khi
    # model tìm đúng cụm gần đúng nhưng sai ranh giới.
    boundary_errors = []

    for predicted_span in fp:
        for gold_span in fn:
            if spans_overlap(predicted_span, gold_span):
                boundary_errors.append({
                    "gold": gold_span,
                    "predicted": predicted_span,
                })

    if not gold and not predicted:
        error_type = "TRUE_NEGATIVE"
    elif not fp and not fn:
        error_type = "CORRECT"
    elif boundary_errors:
        error_type = "BOUNDARY_ERROR"
    elif fp and fn:
        error_type = "FP_AND_FN"
    elif fp:
        error_type = "FALSE_POSITIVE"
    else:
        error_type = "FALSE_NEGATIVE"

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "boundary_errors": boundary_errors,
        "error_type": error_type,
    }


def calculate_metrics(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def save_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(
        path,
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze PhoBERT Opinion Extraction errors"
    )

    parser.add_argument(
        "--test",
        default=str(DEFAULT_TEST_PATH),
    )

    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
    )

    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_DIR),
    )

    args = parser.parse_args()

    test_path = Path(args.test)
    output_dir = Path(args.output)

    if not test_path.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {test_path}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("===== LOADING TEST DATA =====")
    records = load_jsonl(test_path)
    print(f"Total comments: {len(records)}")

    print("\n===== LOADING MODEL =====")
    predictor = OpinionPredictor(args.model)

    comment_rows = []
    span_rows = []
    error_counts = Counter()

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for index, record in enumerate(records, start=1):
        text = record["text"]

        comment_id = (
            record.get("comment_id")
            or record.get("id")
            or str(index)
        )

        gold = get_gold_opinions(record)
        predicted = predictor.predict(text)["opinions"]

        analysis = analyze_comment(gold, predicted)

        tp = analysis["tp"]
        fp = analysis["fp"]
        fn = analysis["fn"]

        total_tp += len(tp)
        total_fp += len(fp)
        total_fn += len(fn)

        error_counts[analysis["error_type"]] += 1

        comment_rows.append({
            "comment_id": comment_id,
            "text": text,
            "gold_opinions": " | ".join(
                span_to_string(s) for s in gold
            ),
            "predicted_opinions": " | ".join(
                span_to_string(s) for s in predicted
            ),
            "tp": len(tp),
            "fp": len(fp),
            "fn": len(fn),
            "error_type": analysis["error_type"],
        })

        for span in tp:
            span_rows.append({
                "comment_id": comment_id,
                "text": text,
                "gold": span_to_string(span),
                "predicted": span_to_string(span),
                "error_type": "TP",
            })

        for span in fp:
            overlaps = [
                s for s in fn if spans_overlap(span, s)
            ]

            span_rows.append({
                "comment_id": comment_id,
                "text": text,
                "gold": " | ".join(
                    span_to_string(s) for s in overlaps
                ),
                "predicted": span_to_string(span),
                "error_type": (
                    "BOUNDARY_FP" if overlaps else "FP"
                ),
            })

        for span in fn:
            overlaps = [
                s for s in fp if spans_overlap(span, s)
            ]

            span_rows.append({
                "comment_id": comment_id,
                "text": text,
                "gold": span_to_string(span),
                "predicted": " | ".join(
                    span_to_string(s) for s in overlaps
                ),
                "error_type": (
                    "BOUNDARY_FN" if overlaps else "FN"
                ),
            })

        if index % 20 == 0 or index == len(records):
            print(
                f"Processed {index}/{len(records)} comments"
            )

    metrics = calculate_metrics(
        total_tp,
        total_fp,
        total_fn,
    )

    comment_csv = output_dir / "comment_analysis.csv"
    span_csv = output_dir / "span_errors.csv"
    summary_json = output_dir / "summary.json"

    save_csv(
        comment_csv,
        comment_rows,
        [
            "comment_id",
            "text",
            "gold_opinions",
            "predicted_opinions",
            "tp",
            "fp",
            "fn",
            "error_type",
        ],
    )

    save_csv(
        span_csv,
        span_rows,
        [
            "comment_id",
            "text",
            "gold",
            "predicted",
            "error_type",
        ],
    )

    summary = {
        "total_comments": len(records),
        "gold_spans": total_tp + total_fn,
        "predicted_spans": total_tp + total_fp,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "comment_error_counts": dict(error_counts),
    }

    with open(
        summary_json,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n===== ERROR ANALYSIS =====")
    print(f"Total comments: {len(records)}")
    print(f"Gold spans: {total_tp + total_fn}")
    print(f"Predicted spans: {total_tp + total_fp}")
    print(f"TP: {total_tp}")
    print(f"FP: {total_fp}")
    print(f"FN: {total_fn}")

    print(f"\nPrecision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")

    print("\n===== COMMENT ERROR TYPES =====")
    for error_type, count in sorted(error_counts.items()):
        print(f"{error_type}: {count}")

    print("\n===== OUTPUT FILES =====")
    print(comment_csv)
    print(span_csv)
    print(summary_json)


if __name__ == "__main__":
    main()
