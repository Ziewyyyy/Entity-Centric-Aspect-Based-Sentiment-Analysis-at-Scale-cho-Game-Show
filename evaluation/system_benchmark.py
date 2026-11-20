
import csv
import json
import time
import statistics
from pathlib import Path

import torch

from serving.app import SentimentAnalysisPipeline


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "data" / "labeled" / "entity" / "test.jsonl"
OUTPUT_DIR = ROOT / "data" / "analysis" / "evaluation"

MAX_SAMPLES = 100
WARMUP_SAMPLES = 5


def percentile(values, percent):
    if not values:
        return 0.0

    sorted_values = sorted(values)

    if len(sorted_values) == 1:
        return sorted_values[0]

    position = (len(sorted_values) - 1) * percent / 100
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)

    weight = position - lower

    return (
        sorted_values[lower] * (1 - weight)
        + sorted_values[upper] * weight
    )


def load_comments():
    comments = []

    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            record = json.loads(line)

            text = record.get("text")

            if isinstance(text, str) and text.strip():
                comments.append(text.strip())

    return comments[:MAX_SAMPLES]


def synchronize_gpu():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    print("=" * 65)
    print("SYSTEM BENCHMARK - ENTITY-CENTRIC ABSA")
    print("=" * 65)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1] Loading test comments...")

    comments = load_comments()

    if not comments:
        raise RuntimeError(
            f"No valid comments found in {INPUT_FILE}"
        )

    print(f"Number of comments: {len(comments)}")

    print("\n[2] Loading pipeline...")

    synchronize_gpu()
    load_start = time.perf_counter()

    pipeline = SentimentAnalysisPipeline()

    synchronize_gpu()
    load_time = time.perf_counter() - load_start

    print(f"Model loading time: {load_time:.2f} seconds")

    print("\n[3] Warm-up...")

    warmup_count = min(WARMUP_SAMPLES, len(comments))

    for i in range(warmup_count):
        try:
            pipeline.predict(comments[i])
        except Exception as error:
            print(f"Warm-up error: {error}")

    synchronize_gpu()

    print(f"Warm-up completed: {warmup_count} comments")

    print("\n[4] Benchmarking...")

    results = []
    errors = []

    for index, comment in enumerate(comments, start=1):
        try:
            synchronize_gpu()
            start = time.perf_counter()

            pipeline.predict(comment)

            synchronize_gpu()
            elapsed = time.perf_counter() - start

            results.append({
                "index": index,
                "latency_seconds": elapsed,
                "latency_ms": elapsed * 1000,
                "text": comment,
            })

            print(
                f"[{index}/{len(comments)}] "
                f"{elapsed * 1000:.2f} ms"
            )

        except Exception as error:
            errors.append({
                "index": index,
                "text": comment,
                "error": str(error),
            })

            print(f"[{index}/{len(comments)}] ERROR: {error}")

    if not results:
        raise RuntimeError("No comments were successfully benchmarked.")

    latencies = [
        item["latency_seconds"]
        for item in results
    ]

    total_time = sum(latencies)

    metrics = {
        "total_comments": len(comments),
        "successful_comments": len(results),
        "failed_comments": len(errors),
        "model_loading_seconds": load_time,
        "total_inference_seconds": total_time,
        "latency": {
            "mean_ms": statistics.mean(latencies) * 1000,
            "median_ms": statistics.median(latencies) * 1000,
            "min_ms": min(latencies) * 1000,
            "max_ms": max(latencies) * 1000,
            "p95_ms": percentile(latencies, 95) * 1000,
            "p99_ms": percentile(latencies, 99) * 1000,
        },
        "throughput_comments_per_second": (
            len(results) / total_time if total_time > 0 else 0
        ),
        "device": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "CPU"
        ),
        "errors": errors,
    }

    print("\n[5] Saving benchmark results...")

    json_path = OUTPUT_DIR / "system_benchmark.json"
    csv_path = OUTPUT_DIR / "system_benchmark.csv"

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(
            metrics,
            file,
            ensure_ascii=False,
            indent=4,
        )

    with open(csv_path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "index",
                "latency_seconds",
                "latency_ms",
                "text",
            ],
        )

        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 65)
    print("BENCHMARK COMPLETED")
    print("=" * 65)

    print(f"Device: {metrics['device']}")
    print(f"Total comments: {metrics['total_comments']}")
    print(f"Successful: {metrics['successful_comments']}")
    print(f"Failed: {metrics['failed_comments']}")

    print(f"\nModel loading: {load_time:.2f} seconds")
    print(f"Total inference: {total_time:.2f} seconds")

    print("\nLATENCY")
    for name, value in metrics["latency"].items():
        print(f"{name}: {value:.2f} ms")

    print(
        "\nThroughput: "
        f"{metrics['throughput_comments_per_second']:.2f} comments/second"
    )

    print(f"\nJSON report: {json_path}")
    print(f"CSV report: {csv_path}")


if __name__ == "__main__":
    main()
