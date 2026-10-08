
import os
import json
import time
import argparse
from pathlib import Path

from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

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


def get_video_info(video_id):
    response = youtube.videos().list(
        part="snippet,statistics",
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


def crawl_video(video, limit=500):
    video_id = video["video_id"]
    info = get_video_info(video_id)

    print(f"\nCrawling: {info['video_title']}")

    comments = []
    page_token = None

    while len(comments) < limit:
        try:
            response = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=min(100, limit - len(comments)),
                pageToken=page_token,
                order="time",
                textFormat="plainText"
            ).execute()

        except HttpError as error:
            print(f"API error: {error}")
            break

        for item in response.get("items", []):
            comment = item["snippet"]["topLevelComment"]
            snippet = comment["snippet"]

            record = {
                "comment_id": comment["id"],
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

        print(f"Collected: {len(comments)}/{limit}")

        page_token = response.get("nextPageToken")

        if not page_token:
            break

        time.sleep(0.3)

    return comments


def save_jsonl(comments, video_id):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / f"{video_id}.jsonl"

    with open(output_path, "w", encoding="utf-8") as file:
        for comment in comments:
            file.write(
                json.dumps(comment, ensure_ascii=False) + "\n"
            )

    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()

    if args.limit < 1:
        parser.error("--limit must be greater than 0")

    with open(VIDEO_FILE, encoding="utf-8") as file:
        config = json.load(file)

    total = 0

    for video in config["videos"]:
        try:
            comments = crawl_video(video, args.limit)

            save_jsonl(comments, video["video_id"])

            total += len(comments)

        except (HttpError, ValueError) as error:
            print(f"Skipping video: {error}")

    print("\n==============================")
    print(f"TOTAL COMMENTS: {total}")
    print("==============================")


if __name__ == "__main__":
    main()
