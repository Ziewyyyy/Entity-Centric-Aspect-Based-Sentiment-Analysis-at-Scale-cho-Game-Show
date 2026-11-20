"""Publish crawler JSONL records to Kafka's raw-comment topic."""

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from confluent_kafka import KafkaException, Producer


ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger("youtube-producer")


def normalize_comment(record):
    """Validate the wire schema while preserving the original comment text."""
    if not isinstance(record, dict):
        raise ValueError("Each record must be a JSON object")

    message = {}
    for field in ("comment_id", "video_id", "text", "published_at"):
        value = record.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be a non-empty string")
        message[field] = value

    message["comment_id"] = message["comment_id"].strip()
    message["video_id"] = message["video_id"].strip()
    published_at = datetime.fromisoformat(
        message["published_at"].strip().replace("Z", "+00:00")
    )
    if published_at.utcoffset() is None:
        raise ValueError("published_at must include a timezone")
    message["published_at"] = (
        published_at.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    )

    # Optional context fields: missing or null values are sent as null.
    for field in ("video_title", "show"):
        value = record.get(field)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{field} must be a string or null")
        message[field] = value

    like_count = record.get("like_count")
    if like_count is not None and (
        isinstance(like_count, bool) or not isinstance(like_count, int) or like_count < 0
    ):
        raise ValueError("like_count must be a non-negative integer or null")
    message["like_count"] = like_count

    # Reject lone Unicode surrogates, including escaped ones in otherwise valid JSON.
    for value in message.values():
        if isinstance(value, str):
            value.encode("utf-8")
    return message


def enqueue_message(producer, topic, message, callback, timeout_seconds):
    """Wait briefly for queue capacity; broker retries are handled by the client."""
    key = message["comment_id"].encode("utf-8")
    value = json.dumps(message, ensure_ascii=False, allow_nan=False).encode("utf-8")
    deadline = time.monotonic() + timeout_seconds

    while True:
        try:
            producer.produce(topic, key=key, value=value, on_delivery=callback)
            return
        except BufferError:
            if time.monotonic() >= deadline:
                raise TimeoutError("Producer queue remained full") from None
            producer.poll(0.1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=ROOT / "data" / "raw" / "performance",
        help="JSONL file or directory containing JSONL files",
    )
    parser.add_argument("--bootstrap-servers", default="localhost:9092")
    parser.add_argument("--topic", default="raw-comment")
    parser.add_argument("--delivery-timeout-ms", type=int, default=60000)
    parser.add_argument("--log-file", type=Path, default=ROOT / "logs" / "producer.log")
    args = parser.parse_args(argv)
    if args.delivery_timeout_ms < 1000:
        parser.error("--delivery-timeout-ms must be at least 1000")

    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.log_file, encoding="utf-8"),
        ],
        force=True,
    )

    if args.input.is_file():
        files = [args.input]
    elif args.input.is_dir():
        files = sorted(args.input.glob("*.jsonl"))
    else:
        files = []
    if not files:
        LOGGER.error("INPUT_ERROR no JSONL files found at %s", args.input)
        return 1

    try:
        producer = Producer({
            "bootstrap.servers": args.bootstrap_servers,
            "client.id": "youtube-comments-producer",
            "acks": "all",
            "enable.idempotence": True,
            "retries": 10,
            "retry.backoff.ms": 500,
            "delivery.timeout.ms": args.delivery_timeout_ms,
        }, logger=LOGGER)
    except (KafkaException, ValueError) as error:
        LOGGER.error("PRODUCER_INIT_FAILED error=%s", error)
        return 1

    counts = {"read": 0, "success": 0, "invalid": 0, "failed": 0, "file_errors": 0}

    def delivery_report(error, kafka_message):
        comment_id = kafka_message.key().decode("utf-8")
        if error is not None:
            counts["failed"] += 1
            LOGGER.error("SEND_FAILED comment_id=%r error=%s", comment_id, error)
        else:
            counts["success"] += 1
            LOGGER.info(
                "SEND_OK comment_id=%r topic=%s partition=%s offset=%s",
                comment_id, kafka_message.topic(), kafka_message.partition(),
                kafka_message.offset(),
            )

    interrupted = False
    LOGGER.info("START input=%s topic=%s bootstrap=%s", args.input, args.topic, args.bootstrap_servers)
    try:
        for path in files:
            LOGGER.info("READ_FILE path=%s", path)
            try:
                # Decode per line so a malformed UTF-8 record cannot skip the file.
                with path.open("rb") as source:
                    for line_number, raw_line in enumerate(source, start=1):
                        producer.poll(0)
                        if not raw_line.strip():
                            continue
                        counts["read"] += 1
                        try:
                            record = json.loads(raw_line.decode("utf-8"))
                            message = normalize_comment(record)
                        except (ValueError, OverflowError) as error:
                            counts["invalid"] += 1
                            LOGGER.error("INVALID file=%s line=%s error=%s", path, line_number, error)
                            continue

                        try:
                            enqueue_message(
                                producer, args.topic, message, delivery_report,
                                timeout_seconds=min(10, args.delivery_timeout_ms / 1000),
                            )
                        except (KafkaException, TimeoutError, ValueError) as error:
                            counts["failed"] += 1
                            LOGGER.error(
                                "ENQUEUE_FAILED file=%s line=%s comment_id=%r error=%s",
                                path, line_number, message["comment_id"], error,
                            )
            except OSError as error:
                counts["file_errors"] += 1
                LOGGER.error("FILE_ERROR path=%s error=%s", path, error)
    except KeyboardInterrupt:
        interrupted = True
        LOGGER.warning("INTERRUPTED waiting for queued messages")
    finally:
        pending = producer.flush(args.delivery_timeout_ms / 1000 + 5)
        LOGGER.info(
            "SUMMARY read=%s success=%s invalid=%s failed=%s pending=%s file_errors=%s interrupted=%s",
            counts["read"], counts["success"], counts["invalid"], counts["failed"],
            pending, counts["file_errors"], interrupted,
        )

    if not counts["read"]:
        LOGGER.error("INPUT_ERROR input files contain no records")
    return int(bool(
        counts["invalid"] or counts["failed"] or counts["file_errors"]
        or pending or interrupted or not counts["read"]
    ))


if __name__ == "__main__":
    raise SystemExit(main())
