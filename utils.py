"""
utils.py — helper functions for YT Summarizer
Compatible with Python 3.11+
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import textwrap
from typing import Any
from urllib.parse import parse_qs, urlparse

import requests
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    CouldNotRetrieveTranscript,
    NoTranscriptFound,
    RequestBlocked,
    TranscriptsDisabled,
    VideoUnavailable,
)


# ── Video ID extraction ───────────────────────────────────────────────────────

def extract_video_id(url: str) -> str | None:
    url = url.strip()
    parsed = urlparse(url)

    if parsed.netloc in ("youtu.be", "www.youtu.be"):
        vid = parsed.path.lstrip("/").split("/")[0]
        return vid if _valid_id(vid) else None

    if "youtube.com" in parsed.netloc:
        qs = parse_qs(parsed.query)
        if "v" in qs:
            vid = qs["v"][0]
            return vid if _valid_id(vid) else None
        match = re.search(r"(?:embed|shorts|v)/([A-Za-z0-9_-]{11})", parsed.path)
        if match:
            return match.group(1)

    match = re.fullmatch(r"[A-Za-z0-9_-]{11}", url)
    if match:
        return url
    return None


def _valid_id(vid: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9_-]{11}", vid))


# ── Duration / views helpers ──────────────────────────────────────────────────

def _parse_iso_duration(iso: str) -> str:
    """PT1H23M45S  →  1:23:45"""
    m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", iso)
    if not m:
        return iso
    h, mn, s = int(m.group(1) or 0), int(m.group(2) or 0), int(m.group(3) or 0)
    return f"{h}:{mn:02d}:{s:02d}" if h else f"{mn}:{s:02d}"


def _fmt_views(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n/1_000_000_000:.1f}B"
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)


# ── Video metadata ────────────────────────────────────────────────────────────

def fetch_video_metadata(video_id: str) -> dict[str, Any] | None:
    """Try scraping the watch page first; fall back to oEmbed."""
    return _scrape_metadata(video_id) or _oembed_metadata(video_id)


def _scrape_metadata(video_id: str) -> dict[str, Any] | None:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        resp = requests.get(
            f"https://www.youtube.com/watch?v={video_id}",
            headers=headers, timeout=12,
        )
        if resp.status_code != 200:
            return None
        html = resp.text

        # Title — from og:title meta tag (most reliable)
        title = "Unknown"
        og = re.search(r'<meta property="og:title" content="([^"]+)"', html)
        if og:
            title = og.group(1)
        else:
            t = re.search(r'"title"\s*:\s*\{"runs":\[.*?"text"\s*:\s*"([^"]+)"', html)
            if t:
                title = t.group(1)

        # Author
        author = "Unknown"
        a = re.search(r'"ownerChannelName"\s*:\s*"([^"]+)"', html)
        if a:
            author = a.group(1)
        else:
            a2 = re.search(r'"author"\s*:\s*"([^"]+)"', html)
            if a2:
                author = a2.group(1)

        # Duration (ISO 8601)
        duration = "N/A"
        d = re.search(r'"lengthSeconds"\s*:\s*"(\d+)"', html)
        if d:
            secs = int(d.group(1))
            h, r = divmod(secs, 3600)
            mn, s = divmod(r, 60)
            duration = f"{h}:{mn:02d}:{s:02d}" if h else f"{mn}:{s:02d}"
        else:
            d2 = re.search(r'"duration"\s*:\s*"(PT[^"]+)"', html)
            if d2:
                duration = _parse_iso_duration(d2.group(1))

        # Views
        views = "N/A"
        v = re.search(r'"viewCount"\s*:\s*"(\d+)"', html)
        if v:
            views = _fmt_views(int(v.group(1)))

        # Thumbnail — prefer maxresdefault, fallback to hqdefault
        thumbnail = f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"
        th = re.search(r'<meta property="og:image" content="([^"]+)"', html)
        if th:
            thumbnail = th.group(1)

        return {
            "title": title,
            "author": author,
            "thumbnail": thumbnail,
            "duration": duration,
            "views": views,
        }
    except Exception:
        return None


def _oembed_metadata(video_id: str) -> dict[str, Any] | None:
    try:
        resp = requests.get(
            "https://www.youtube.com/oembed",
            params={"url": f"https://www.youtube.com/watch?v={video_id}", "format": "json"},
            timeout=8,
        )
        if resp.status_code == 200:
            data = resp.json()
            return {
                "title":     data.get("title", "Unknown"),
                "author":    data.get("author_name", "Unknown"),
                "thumbnail": data.get("thumbnail_url", f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"),
                "duration":  "N/A",
                "views":     "N/A",
            }
    except Exception:
        pass
    return None


# ── Transcript fetching ───────────────────────────────────────────────────────

def fetch_transcript(
    video_id: str,
    include_timestamps: bool = False,
) -> tuple[str, str | None]:
    """
    Try youtube-transcript-api first; if blocked, fall back to yt-dlp.
    Returns (text, error).  error is None on success.
    """
    text, err = _transcript_via_api(video_id, include_timestamps)
    if text:
        return text, None

    # Fallback: yt-dlp (handles IP blocks on cloud servers)
    text2, err2 = _transcript_via_ytdlp(video_id, include_timestamps)
    if text2:
        return text2, None

    return "", err2 or err or "Could not retrieve transcript."


def _transcript_via_api(
    video_id: str, include_timestamps: bool
) -> tuple[str, str | None]:
    try:
        ytt = YouTubeTranscriptApi()
        tlist = ytt.list(video_id)
        try:
            t = tlist.find_manually_created_transcript(["en"])
        except NoTranscriptFound:
            try:
                t = tlist.find_generated_transcript(["en"])
            except NoTranscriptFound:
                t = next(iter(tlist))
        segs = t.fetch()
        return _segs_to_text(segs, include_timestamps), None
    except (TranscriptsDisabled, VideoUnavailable, NoTranscriptFound) as e:
        return "", str(e)
    except (RequestBlocked, CouldNotRetrieveTranscript):
        return "", "blocked"
    except Exception as e:
        return "", f"Unexpected error: {e}"


def _transcript_via_ytdlp(
    video_id: str, include_timestamps: bool
) -> tuple[str, str | None]:
    """Download auto-subs via yt-dlp CLI and parse VTT."""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_tmpl = os.path.join(tmpdir, "%(id)s.%(ext)s")
            cmd = [
                "yt-dlp", "--skip-download",
                "--write-auto-sub", "--sub-lang", "en",
                "--sub-format", "vtt",
                "--output", out_tmpl,
                "--quiet", "--no-warnings",
                f"https://www.youtube.com/watch?v={video_id}",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            vtt_file = next(
                (os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if f.endswith(".vtt")),
                None,
            )
            if not vtt_file:
                stderr = result.stderr[:300] if result.stderr else "no subtitle file created"
                return "", f"yt-dlp subtitles unavailable: {stderr}"

            with open(vtt_file, encoding="utf-8") as f:
                return _parse_vtt(f.read(), include_timestamps), None

    except FileNotFoundError:
        return "", "yt-dlp not found — add 'yt-dlp' to requirements.txt."
    except subprocess.TimeoutExpired:
        return "", "yt-dlp timed out."
    except Exception as e:
        return "", f"yt-dlp error: {e}"


def _parse_vtt(vtt: str, include_timestamps: bool) -> str:
    """Parse WebVTT into clean text, deduplicating repeated lines."""
    segments: list[tuple[str, str]] = []
    current_ts = ""
    current_lines: list[str] = []

    for line in vtt.splitlines():
        line = line.strip()
        if "-->" in line:
            if current_lines:
                segments.append((current_ts, " ".join(current_lines)))
                current_lines = []
            current_ts = line.split("-->")[0].strip()
        elif line and not line.startswith("WEBVTT") and not line.isdigit() and "align:" not in line:
            clean = re.sub(r"<[^>]+>", "", line).strip()
            if clean:
                current_lines.append(clean)

    if current_lines:
        segments.append((current_ts, " ".join(current_lines)))

    # Deduplicate consecutive identical segments (common in auto-subs)
    deduped: list[tuple[str, str]] = []
    prev = ""
    for ts, text in segments:
        if text != prev:
            deduped.append((ts, text))
            prev = text

    if include_timestamps:
        return "\n".join(f"[{ts.split('.')[0]}] {text}" for ts, text in deduped)
    return " ".join(text for _, text in deduped)


def _segs_to_text(segments: list[Any], include_timestamps: bool) -> str:
    if include_timestamps:
        return "\n".join(f"[{_fmt_ts(seg.start)}] {seg.text}" for seg in segments)
    return " ".join(seg.text for seg in segments)


def _fmt_ts(seconds: float) -> str:
    seconds = int(seconds)
    h, r = divmod(seconds, 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


# ── Gemini ────────────────────────────────────────────────────────────────────

_GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.0-flash:generateContent"
)

_STYLE_INSTRUCTIONS = {
    "Concise":         "Provide a concise summary in 3-5 short paragraphs.",
    "Detailed":        "Provide a thorough, detailed summary covering all major topics discussed.",
    "Bullet Points":   "Summarize using clearly organized bullet points grouped by topic.",
    "Executive Brief": (
        "Write an executive brief: one-line TL;DR, followed by 3-5 high-impact insights, "
        "and a recommended action or takeaway."
    ),
}


def _call_gemini(prompt: str, api_key: str, max_tokens: int = 1024) -> str:
    resp = requests.post(
        f"{_GEMINI_ENDPOINT}?key={api_key}",
        headers={"Content-Type": "application/json"},
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.4, "topP": 0.9},
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError) as e:
        raise ValueError(f"Unexpected Gemini response: {data}") from e


def summarize_with_gemini(
    transcript: str, api_key: str, style: str = "Concise",
    language: str = "English", max_tokens: int = 1024,
) -> str:
    instr = _STYLE_INSTRUCTIONS.get(style, _STYLE_INSTRUCTIONS["Concise"])
    prompt = textwrap.dedent(f"""
        You are an expert content analyst summarizing a YouTube video transcript.
        Style: {instr}
        Output language: {language}

        Transcript:
        {transcript[:30_000]}

        Provide only the summary with no preamble.
    """).strip()
    return _call_gemini(prompt, api_key, max_tokens)


def generate_key_points(transcript: str, api_key: str, max_tokens: int = 1024) -> list[str] | str:
    prompt = textwrap.dedent(f"""
        Extract the 7 most important key points from this YouTube transcript.
        Respond ONLY with a valid JSON array of strings.
        Do not include any other text.

        Transcript:
        {transcript[:30_000]}
    """).strip()
    raw = _call_gemini(prompt, api_key, max_tokens)
    try:
        clean = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
        pts = json.loads(clean)
        if isinstance(pts, list):
            return [str(p) for p in pts]
    except (json.JSONDecodeError, ValueError):
        pass
    return raw


def generate_quiz(transcript: str, api_key: str, n_questions: int = 5) -> list[dict] | str:
    prompt = textwrap.dedent(f"""
        Create {n_questions} comprehension questions from this YouTube transcript.
        Respond ONLY with a valid JSON array:
        [{{"question":"...","answer":"...","explanation":"..."}}]
        No other text or markdown.

        Transcript:
        {transcript[:25_000]}
    """).strip()
    raw = _call_gemini(prompt, api_key, 1024)
    try:
        clean = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
        items = json.loads(clean)
        if isinstance(items, list):
            return items
    except (json.JSONDecodeError, ValueError):
        pass
    return raw


# ── Export ────────────────────────────────────────────────────────────────────

def export_summary_as_txt(title: str, summary: str, key_points: list[str] | str) -> str:
    sep = "=" * 60
    lines = [sep, f"YouTube Video Summary: {title}", sep, "", "SUMMARY", "-------", summary, "", "KEY POINTS", "----------"]
    if isinstance(key_points, list):
        for i, pt in enumerate(key_points, 1):
            lines.append(f"{i}. {pt}")
    else:
        lines.append(key_points)
    lines += ["", sep]
    return "\n".join(lines)
