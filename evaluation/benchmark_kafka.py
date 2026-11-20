
import argparse
import csv
import json
import logging
import math
import statistics
import time

from pathlib import Path
from kafka import KafkaProducer
from kafka.errors import KafkaError


# =====================================
# CONFIGURATION
# =====================================

ROOT = Path(__file__).resolve().parents[1]

BOOTSTRAP_SERVERS = "127.0.0.1:9092"
DEFAULT_TOPIC = "raw-comment-benchmark"

DEFAULT_INPUT_DIR = ROOT / "data" / "raw" / "performance"
OUTPUT_DIR = ROOT / "data" / "analysis" / "benchmark"

REQUIRED_FIELDS = [
    "comment_id",
    "video_id",
    "text",
    "published_at"
]

BENCHMARK_SIZES = {
    100: 1,
    300: 3,
    900: 9
}


# =====================================
# LOGGING
# =====================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# =====================================
# PRODUCER
# Same configuration as original producer
# =====================================

def create_producer():

    return KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,

        key_serializer=lambda key: key.encode("utf-8"),

        value_serializer=lambda value: json.dumps(
            value,
            ensure_ascii=False
        ).encode("utf-8"),

        acks="all",
        retries=5,
        max_in_flight_requests_per_connection=1,
        request_timeout_ms=30000,
        max_block_ms=30000
    )


# =====================================
# DATA VALIDATION
# Same normalization as original producer
# =====================================

def normalize_comment(comment):

    for field in REQUIRED_FIELDS:
        if field not in comment or comment[field] is None:
            raise ValueError(f"Missing required field: {field}")

    normalized = {
        "comment_id": str(comment["comment_id"]).strip(),
        "video_id": str(comment["video_id"]).strip(),
        "text": str(comment["text"]),
        "published_at": str(comment["published_at"])
    }

    for field in REQUIRED_FIELDS:
        if not normalized[field]:
            raise ValueError(f"Empty required field: {field}")

    return normalized


# =====================================
# PERCENTILE
# =====================================

def percentile(values, percent):

    if not values:
        return 0.0

    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * percent / 100

    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return ordered[lower]

    weight = position - lower

    return (
        ordered[lower] * (1 - weight)
        + ordered[upper] * weight
    )


# =====================================
# LOAD BENCHMARK FILES
# =====================================

def get_files(input_dir):

    files = sorted(input_dir.glob("*.jsonl"))

    if len(files) < 9:
        raise FileNotFoundError(
            f"Expected at least 9 JSONL files, found {len(files)}"
        )

    return files[:9]


# =====================================
# RUN ONE BENCHMARK
# =====================================

def run_benchmark(files, topic, run_number, expected_count):

    producer = None

    success = 0
    failed = 0
    latencies_ms = []

    total_start = None

    try:
        producer = create_producer()

        # Exclude producer initialization from measurement.
        producer.partitions_for(topic)

        logger.info(
            "RUN %s | expected=%s | topic=%s",
            run_number,
            expected_count,
            topic
        )

        total_start = time.perf_counter()

        for file_path in files:

            logger.info("Reading %s", file_path.name)

            with file_path.open("r", encoding="utf-8") as file:

                for line_number, line in enumerate(file, start=1):

                    if not line.strip():
                        continue

                    try:
                        comment = json.loads(line)
                        normalized = normalize_comment(comment)

                        message_start = time.perf_counter()

                        future = producer.send(
                            topic,
                            key=normalized["comment_id"],
                            value=normalized
                        )

                        future.get(timeout=30)

                        message_end = time.perf_counter()

                        latencies_ms.append(
                            (message_end - message_start) * 1000
                        )

                        success += 1

                    except (json.JSONDecodeError, ValueError) as error:
                        failed += 1
                        logger.error(
                            "INVALID MESSAGE | file=%s | line=%s | %s",
                            file_path.name,
                            line_number,
                            error
                        )

                    except Exception as error:
                        failed += 1
                        logger.error(
                            "SEND FAILED | file=%s | line=%s | %s",
                            file_path.name,
                            line_number,
                            error
                        )

        producer.flush(timeout=30)

        total_end = time.perf_counter()

        elapsed_seconds = total_end - total_start

        throughput = (
            success / elapsed_seconds
            if elapsed_seconds > 0 else 0
        )

        result = {
            "run": run_number,
            "expected": expected_count,
            "success": success,
            "failed": failed,
            "elapsed_seconds": round(elapsed_seconds, 6),
            "throughput_msg_per_sec": round(throughput, 4),
            "avg_latency_ms": round(
                statistics.mean(latencies_ms), 4
            ) if latencies_ms else 0,
            "p50_latency_ms": round(
                percentile(latencies_ms, 50), 4
            ),
            "p95_latency_ms": round(
                percentile(latencies_ms, 95), 4
            ),
            "p99_latency_ms": round(
                percentile(latencies_ms, 99), 4
            )
        }

        return result

    finally:
        if producer is not None:
            producer.close(timeout=30)


# =====================================
# SAVE RESULTS
# =====================================

def save_results(results):

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    json_path = OUTPUT_DIR / "kafka_benchmark_results.json"
    csv_path = OUTPUT_DIR / "kafka_benchmark_results.csv"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(
            results,
            file,
            ensure_ascii=False,
            indent=4
        )

    csv_fields = [
        "dataset_size",
        "run",
        "expected",
        "success",
        "failed",
        "elapsed_seconds",
        "throughput_msg_per_sec",
        "avg_latency_ms",
        "p50_latency_ms",
        "p95_latency_ms",
        "p99_latency_ms"
    ]

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=csv_fields
        )

        writer.writeheader()

        for size, runs in results.items():
            for result in runs:
                writer.writerow({
                    "dataset_size": size,
                    **result
                })

    logger.info("JSON saved: %s", json_path)
    logger.info("CSV saved: %s", csv_path)


# =====================================
# PRINT SUMMARY
# =====================================

def print_summary(results):

    print("\n" + "=" * 90)
    print("KAFKA BENCHMARK SUMMARY")
    print("=" * 90)

    header = (
        f"{'Messages':>10} "
        f"{'Runs':>6} "
        f"{'Success':>9} "
        f"{'Failed':>8} "
        f"{'Time(s)':>10} "
        f"{'Throughput':>13} "
        f"{'Avg(ms)':>10} "
        f"{'P95(ms)':>10} "
        f"{'P99(ms)':>10}"
    )

    print(header)
    print("-" * 90)

    for size, runs in results.items():

        def average(field):
            return statistics.mean(
                run[field] for run in runs
            )

        print(
            f"{size:>10} "
            f"{len(runs):>6} "
            f"{average('success'):>9.1f} "
            f"{average('failed'):>8.1f} "
            f"{average('elapsed_seconds'):>10.3f} "
            f"{average('throughput_msg_per_sec'):>13.2f} "
            f"{average('avg_latency_ms'):>10.2f} "
            f"{average('p95_latency_ms'):>10.2f} "
            f"{average('p99_latency_ms'):>10.2f}"
        )

    print("=" * 90)


# =====================================
# MAIN
# =====================================

def main():

    parser = argparse.ArgumentParser(
        description="Kafka synchronous producer benchmark"
    )

    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_DIR)
    )

    parser.add_argument(
        "--topic",
        default=DEFAULT_TOPIC
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=3
    )

    args = parser.parse_args()

    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")

    input_dir = Path(args.input)

    if not input_dir.exists():
        raise FileNotFoundError(input_dir)

    files = get_files(input_dir)

    results = {}

    print("\nKafka benchmark configuration")
    print(f"Broker: {BOOTSTRAP_SERVERS}")
    print(f"Topic: {args.topic}")
    print(f"Input: {input_dir}")
    print(f"Repeats: {args.repeats}")
    print(f"Videos available: {len(files)}")

    for size, video_count in BENCHMARK_SIZES.items():

        selected_files = files[:video_count]

        print(f"\n{'=' * 60}")
        print(
            f"BENCHMARK: {size} COMMENTS "
            f"({video_count} VIDEO FILES)"
        )
        print(f"{'=' * 60}")

        results[size] = []

        for run_number in range(1, args.repeats + 1):

            result = run_benchmark(
                files=selected_files,
                topic=args.topic,
                run_number=run_number,
                expected_count=size
            )

            results[size].append(result)

            print(
                f"Run {run_number}: "
                f"success={result['success']}, "
                f"failed={result['failed']}, "
                f"time={result['elapsed_seconds']}s, "
                f"throughput={result['throughput_msg_per_sec']} msg/s"
            )

    save_results(results)
    print_summary(results)


if __name__ == "__main__":
    main()
