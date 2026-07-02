# YouTube Transcript Fetcher

`get_transcript.py` downloads the caption/transcript track for a YouTube video
and saves it as text.

> **Why a script instead of the transcript itself?** The Claude Code web
> environment this was built in has a restricted network policy that only
> allows package registries (pypi/npm). It cannot reach `youtube.com`, so the
> transcript must be fetched from a machine with normal internet access.

## Setup

```bash
pip install youtube-transcript-api
```

## Usage

```bash
# Plain text to stdout
python get_transcript.py https://youtu.be/83fWzQSWB10

# With [mm:ss] timestamps, written to a file
python get_transcript.py 83fWzQSWB10 --timestamps -o transcript.txt

# Prefer a non-English caption track (falls back per your list)
python get_transcript.py <id> --languages es,en
```

Accepts a full URL (`youtu.be/...`, `youtube.com/watch?v=...`, `/shorts/...`,
`/embed/...`) or a bare 11-character video ID.

## The video you asked about

- **ID:** `83fWzQSWB10`
- **Title:** *AI Agents are the new SaaS*
- **Run:** `python get_transcript.py 83fWzQSWB10 --timestamps -o transcript.txt`
