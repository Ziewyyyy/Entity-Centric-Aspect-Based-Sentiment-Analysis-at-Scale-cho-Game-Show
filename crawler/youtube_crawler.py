
import os
import json
import time
import argparse
from pathlib import Path

from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


# ==========================
# CONFIG
# ==========================

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

API_KEY = os.getenv("YOUTUBE_API_KEY")

if not API_KEY:
    raise ValueError("YOUTUBE_API_KEY not found in .env")

youtube = build(
    "youtube",
    "v3",
    developerKey=API_KEY
)

VIDEO_FILE = ROOT / "crawler" / "performance_videos.json"
OUTPUT_DIR = ROOT / "data" / "raw" / "performance"

# Chỉ crawl 2 video mới
NEW_VIDEO_IDS = [
    "qKTqFeRb43w"
]


# ==========================
# GET VIDEO INFO
# ==========================

def get_video_info(video_id):
    response = youtube.videos().list(
        part="snippet",
        id=video_id
    ).execute()

    items = response.get("items", [])

    if not items:
        raise ValueError(f"Video not found: {video_id}")

    snippet = items[0]["snippet"]

    return {
        "video_title": snippet["title"],
        "channel_title": snippet["channelTitle"]
    }


# ==========================
# CRAWL COMMENTS
# ==========================

def crawl_video(video, limit=100):
    video_id = video["video_id"].strip()
    info = get_video_info(video_id)

    print("\n==============================")
    print(f"VIDEO: {info['video_title']}")
    print(f"VIDEO ID: {video_id}")
    print(f"LIMIT: {limit}")
    print("==============================")

    comments = []
    seen_ids = set()
    page_token = None

    while len(comments) < limit:
        try:
            params = {
                "part": "snippet",
                "videoId": video_id,
                "maxResults": min(100, limit - len(comments)),
                "order": "time",
                "textFormat": "plainText"
            }

            if page_token:
                params["pageToken"] = page_token

            response = youtube.commentThreads().list(
                **params
            ).execute()

        except HttpError as error:
            print(f"API error: {error}")
            break

        items = response.get("items", [])

        if not items:
            break

        for item in items:
            comment = item["snippet"]["topLevelComment"]
            snippet = comment["snippet"]
            comment_id = comment["id"]

            if comment_id in seen_ids:
                continue

            seen_ids.add(comment_id)

            record = {
                "comment_id": comment_id,
                "video_id": video_id,
                "video_title": info["video_title"],
                "channel_title": info["channel_title"],
                "show": video["show"],
                "season": video["season"],
                "video_type": video["video_type"],
                "source": "youtube",
                "text": snippet["textDisplay"],
                "like_count": snippet.get("likeCount", 0),
                "published_at": snippet["publishedAt"],
                "updated_at": snippet.get("updatedAt"),
                "parent_id": None
            }

            comments.append(record)

            if len(comments) >= limit:
                break

        print(f"Collected: {len(comments)}/{limit}")

        next_page_token = response.get("nextPageToken")

        if not next_page_token or next_page_token == page_token:
            break

        page_token = next_page_token
        time.sleep(0.3)

    return comments


# ==========================
# SAVE JSONL
# ==========================

def save_jsonl(comments, video_id):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = OUTPUT_DIR / f"{video_id}.jsonl"

    with open(output_path, "w", encoding="utf-8") as file:
        for comment in comments:
            file.write(
                json.dumps(
                    comment,
                    ensure_ascii=False
                ) + "\n"
            )

    print(f"Saved: {output_path}")
    return output_path


# ==========================
# MAIN
# ==========================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum comments per video"
    )

    args = parser.parse_args()

    if args.limit < 1:
        parser.error("--limit must be greater than 0")

    with open(VIDEO_FILE, "r", encoding="utf-8") as file:
        config = json.load(file)

    total = 0
    processed_videos = 0
    processed_ids = set()

    print("\n====================================")
    print("CRAWLING NEW PERFORMANCE VIDEOS")
    print("====================================")

    for video in config["videos"]:
        video_id = video["video_id"].strip()

        # Bỏ qua video cũ
        if video_id not in NEW_VIDEO_IDS:
            continue

        # Bỏ qua video trùng
        if video_id in processed_ids:
            continue

        processed_ids.add(video_id)

        video = {
            **video,
            "video_id": video_id
        }

        try:
            comments = crawl_video(
                video,
                limit=args.limit
            )

            save_jsonl(comments, video_id)

            total += len(comments)
            processed_videos += 1

        except (HttpError, ValueError, KeyError) as error:
            print(f"Skipping video {video_id}: {error}")

    print("\n====================================")
    print("CRAWLING COMPLETED")
    print("====================================")
    print(f"Videos processed: {processed_videos}")
    print(f"Total comments: {total}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("====================================")


if __name__ == "__main__":
    main()
