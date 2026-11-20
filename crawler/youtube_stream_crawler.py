"""Poll YouTube top-level comments and stream new ones to Kafka (near real-time).

Place this file next to kafka_producer.py (it reuses normalize_comment and
enqueue_message from there).

Examples:
    python youtube_stream_crawler.py --video-id qKTqFeRb43w
    python youtube_stream_crawler.py --video-id AAA --video-id BBB --interval 30
    python youtube_stream_crawler.py --video-id qKTqFeRb43w --backfill 200
"""

import argparse
import json
import logging
import os
import signal
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

from confluent_kafka import KafkaException, Producer
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kafka_producer import enqueue_message, normalize_comment  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger("youtube-stream")

MIN_TIME = datetime.min.replace(tzinfo=timezone.utc)
LOOKBACK = timedelta(minutes=2)   # re-check a small window: late-surfacing comments
MAX_SEEN = 2000                   # comment_ids remembered per video
MAX_PAGES = 50                    # safety cap per poll
QUOTA_BACKOFF_SECONDS = 600
stop_event = threading.Event()


# ==========================
# HELPERS
# ==========================

def parse_time(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def error_reason(error):
    try:
        return json.loads(error.content)["error"]["errors"][0]["reason"]
    except Exception:
        return ""


def load_state(path):
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            LOGGER.warning("STATE_UNREADABLE path=%s, starting fresh", path)
    return {}


def save_state(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)  # atomic: a crash never leaves a half-written state


def load_video_config(path):
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    return {v["video_id"].strip(): v for v in config.get("videos", [])}


def get_video_info(youtube, video_id):
    response = youtube.videos().list(part="snippet", id=video_id).execute()
    items = response.get("items", [])
    if not items:
        raise ValueError(f"Video not found: {video_id}")
    snippet = items[0]["snippet"]
    return {"video_title": snippet["title"], "channel_title": snippet["channelTitle"]}


def build_record(top_comment, video_id, info, meta):
    snippet = top_comment["snippet"]
    return {
        "comment_id": top_comment["id"],
        "video_id": video_id,
        "video_title": info["video_title"],
        "channel_title": info["channel_title"],
        "show": meta.get("show"),
        "season": meta.get("season"),
        "video_type": meta.get("video_type"),
        "source": "youtube",
        "text": snippet["textDisplay"],
        "author": snippet.get("authorDisplayName"),
        "like_count": snippet.get("likeCount", 0),
        "published_at": snippet["publishedAt"],
        "updated_at": snippet.get("updatedAt"),
        "parent_id": None,
    }


# ==========================
# FETCH NEW COMMENTS
# ==========================

def fetch_new(youtube, video_id, cutoff, seen, limit=None):
    """Return top-level comments newer than cutoff and not in seen (newest first).

    order="time" is newest-first, so we stop paging as soon as we reach comments
    older than cutoff. Normally this is exactly one API request (1 quota unit).
    """
    found = []
    page_token = None
    for _ in range(MAX_PAGES):
        params = {
            "part": "snippet",
            "videoId": video_id,
            "maxResults": 100,
            "order": "time",
            "textFormat": "plainText",
        }
        if page_token:
            params["pageToken"] = page_token
        response = youtube.commentThreads().list(**params).execute()

        reached_old = False
        for item in response.get("items", []):
            top = item["snippet"]["topLevelComment"]
            if parse_time(top["snippet"]["publishedAt"]) < cutoff:
                reached_old = True
                break
            if top["id"] not in seen:
                found.append(top)
                if limit and len(found) >= limit:
                    return found

        page_token = response.get("nextPageToken")
        if reached_old or not page_token:
            break
    return found


# ==========================
# POLL ONE VIDEO
# ==========================

def poll_video(youtube, producer, args, video_id, video_state, info, meta):
    seen = set(video_state["seen"])

    if video_state["watermark"] is None:
        # First run: either backfill N latest comments, or just mark the current ones as known.
        comments = fetch_new(youtube, video_id, MIN_TIME, seen, limit=args.backfill or 100)
        to_send = comments if args.backfill else []
        newest = max((parse_time(c["snippet"]["publishedAt"]) for c in comments),
                     default=datetime.now(timezone.utc))
        video_state["watermark"] = newest.isoformat()
        video_state["seen"] = [c["id"] for c in comments][-MAX_SEEN:]
        LOGGER.info("INIT video=%s known=%s backfill=%s", video_id, len(comments), len(to_send))
        comments = to_send
    else:
        cutoff = parse_time(video_state["watermark"]) - LOOKBACK
        comments = fetch_new(youtube, video_id, cutoff, seen)

    if not comments:
        return 0

    published = {}
    ok_ids, failed_ids = [], []

    def on_delivery(error, kafka_message):
        comment_id = kafka_message.key().decode("utf-8")
        if error is not None:
            failed_ids.append(comment_id)
            LOGGER.error("SEND_FAILED comment_id=%r error=%s", comment_id, error)
        else:
            ok_ids.append(comment_id)

    enqueued = 0
    for top in reversed(comments):  # oldest -> newest keeps order in the topic
        record = build_record(top, video_id, info, meta)
        try:
            message = normalize_comment(record)
        except (ValueError, OverflowError) as error:
            LOGGER.error("INVALID comment_id=%r error=%s", top["id"], error)
            video_state["seen"].append(top["id"])  # never retry a permanently invalid record
            continue
        try:
            enqueue_message(producer, args.topic, message, on_delivery, timeout_seconds=10)
            published[message["comment_id"]] = message["published_at"]
            enqueued += 1
        except (KafkaException, TimeoutError, ValueError) as error:
            LOGGER.error("ENQUEUE_FAILED comment_id=%r error=%s", top["id"], error)

    pending = producer.flush(30)

    video_state["seen"].extend(ok_ids)
    video_state["seen"] = video_state["seen"][-MAX_SEEN:]
    all_ok = enqueued == len(ok_ids) and not failed_ids and not pending
    if ok_ids and all_ok:
        # Only advance the watermark when everything in this batch was delivered;
        # otherwise unsent comments stay inside the lookback window and are retried.
        newest = max(parse_time(published[cid]) for cid in ok_ids)
        current = parse_time(video_state["watermark"])
        video_state["watermark"] = max(newest, current).isoformat()

    LOGGER.info("POLL video=%s new=%s sent=%s failed=%s pending=%s",
                video_id, len(comments), len(ok_ids), len(failed_ids), pending)
    return len(ok_ids)


# ==========================
# MAIN
# ==========================

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video-id", action="append", required=True,
                        help="Video ID to follow (repeatable)")
    parser.add_argument("--interval", type=float, default=30,
                        help="Seconds between polls (default 30; keep <= 45 for <1 min latency)")
    parser.add_argument("--backfill", type=int, default=0,
                        help="On first run, also send the N latest existing comments (default 0)")
    parser.add_argument("--bootstrap-servers", default="localhost:9092")
    parser.add_argument("--topic", default="raw-comment")
    parser.add_argument("--video-config", type=Path,
                        default=ROOT / "crawler" / "performance_videos.json")
    parser.add_argument("--state-file", type=Path,
                        default=ROOT / "state" / "stream_state.json")
    parser.add_argument("--log-file", type=Path, default=ROOT / "logs" / "stream_crawler.log")
    args = parser.parse_args(argv)
    if args.interval < 5:
        parser.error("--interval must be at least 5 seconds")
    if args.backfill < 0:
        parser.error("--backfill must be >= 0")

    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(args.log_file, encoding="utf-8")],
        force=True,
    )

    load_dotenv(ROOT / ".env")
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        LOGGER.error("YOUTUBE_API_KEY not found in .env")
        return 1

    youtube = build("youtube", "v3", developerKey=api_key)
    configured = load_video_config(args.video_config)

    video_ids = list(dict.fromkeys(v.strip() for v in args.video_id))
    infos, metas = {}, {}
    for video_id in video_ids:
        try:
            infos[video_id] = get_video_info(youtube, video_id)
            metas[video_id] = configured.get(video_id, {})
            LOGGER.info("WATCH video=%s title=%r", video_id, infos[video_id]["video_title"])
        except (HttpError, ValueError, KeyError) as error:
            LOGGER.error("SKIP video=%s error=%s", video_id, error)
    if not infos:
        LOGGER.error("No valid videos to follow")
        return 1

    units_per_day = len(infos) * 86400 / args.interval
    LOGGER.info("QUOTA estimate=%d units/day (default limit 10000)", units_per_day)
    if units_per_day > 9000:
        LOGGER.warning("Estimated usage is close to the daily quota; raise --interval or reduce videos")

    try:
        producer = Producer({
            "bootstrap.servers": args.bootstrap_servers,
            "client.id": "youtube-stream-crawler",
            "acks": "all",
            "enable.idempotence": True,
            "retries": 10,
            "retry.backoff.ms": 500,
            "delivery.timeout.ms": 30000,
            "linger.ms": 20,
        }, logger=LOGGER)
    except (KafkaException, ValueError) as error:
        LOGGER.error("PRODUCER_INIT_FAILED error=%s", error)
        return 1

    state = load_state(args.state_file)
    for video_id in infos:
        state.setdefault(video_id, {"watermark": None, "seen": []})

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop_event.set())

    LOGGER.info("START topic=%s bootstrap=%s interval=%ss videos=%s",
                args.topic, args.bootstrap_servers, args.interval, list(infos))
    total = 0
    try:
        while not stop_event.is_set():
            started = datetime.now(timezone.utc)
            delay = args.interval

            for video_id in list(infos):
                if stop_event.is_set():
                    break
                try:
                    total += poll_video(youtube, producer, args, video_id,
                                        state[video_id], infos[video_id], metas[video_id])
                except HttpError as error:
                    reason = error_reason(error)
                    if reason in ("commentsDisabled", "videoNotFound", "forbidden"):
                        LOGGER.error("DROP video=%s reason=%s", video_id, reason)
                        infos.pop(video_id)
                    elif reason in ("quotaExceeded", "dailyLimitExceeded", "rateLimitExceeded"):
                        LOGGER.error("QUOTA_EXCEEDED reason=%s, backing off %ss",
                                     reason, QUOTA_BACKOFF_SECONDS)
                        delay = QUOTA_BACKOFF_SECONDS
                        break
                    else:
                        LOGGER.error("API_ERROR video=%s reason=%s error=%s",
                                     video_id, reason, error)
                except Exception as error:  # keep the service alive on transient failures
                    LOGGER.exception("POLL_ERROR video=%s error=%s", video_id, error)

            save_state(args.state_file, state)
            if not infos:
                LOGGER.error("No videos left to follow")
                break

            elapsed = (datetime.now(timezone.utc) - started).total_seconds()
            stop_event.wait(max(0.0, delay - elapsed))
    finally:
        pending = producer.flush(30)
        save_state(args.state_file, state)
        LOGGER.info("STOPPED total_sent=%s pending=%s", total, pending)

    return 0 if infos else 1


if __name__ == "__main__":
    raise SystemExit(main())
