#!/usr/bin/env python3
"""Fetch a YouTube video's transcript and save it to a text file.

Usage:
    python get_transcript.py <video-url-or-id> [--timestamps] [-o output.txt]

Examples:
    python get_transcript.py https://youtu.be/83fWzQSWB10
    python get_transcript.py 83fWzQSWB10 --timestamps -o transcript.txt

Requires:
    pip install youtube-transcript-api
"""
import argparse
import re
import sys


def extract_video_id(value: str) -> str:
    """Accept a full URL or a bare 11-char video ID and return the ID."""
    value = value.strip()
    # Bare ID
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", value):
        return value
    # youtu.be/<id> or youtube.com/watch?v=<id> or /embed/<id> etc.
    m = re.search(r"(?:v=|youtu\.be/|/embed/|/shorts/)([A-Za-z0-9_-]{11})", value)
    if m:
        return m.group(1)
    raise ValueError(f"Could not extract a video ID from: {value!r}")


def format_timestamp(seconds: float) -> str:
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch a YouTube transcript.")
    parser.add_argument("video", help="YouTube URL or 11-character video ID")
    parser.add_argument(
        "--timestamps", action="store_true", help="Prefix each line with [mm:ss]"
    )
    parser.add_argument(
        "-o", "--output", help="Write transcript here (default: stdout)"
    )
    parser.add_argument(
        "--languages",
        default="en",
        help="Comma-separated preferred language codes (default: en)",
    )
    args = parser.parse_args()

    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        print(
            "Missing dependency. Install it with:\n"
            "    pip install youtube-transcript-api",
            file=sys.stderr,
        )
        return 1

    video_id = extract_video_id(args.video)
    languages = [c.strip() for c in args.languages.split(",") if c.strip()]

    try:
        entries = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
    except Exception as exc:  # noqa: BLE001 - surface the library's message
        print(f"Failed to fetch transcript for {video_id}: {exc}", file=sys.stderr)
        return 2

    if args.timestamps:
        lines = [f"[{format_timestamp(e['start'])}] {e['text']}" for e in entries]
    else:
        lines = [e["text"] for e in entries]
    text = "\n".join(lines) + "\n"

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"Wrote {len(entries)} lines to {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
