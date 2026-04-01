"""
utils.py — helper functions for YT Summarizer
Compatible with Python 3.13+ (no walrus / match restrictions, standard typing).
"""

from __future__ import annotations

import json
import re
import textwrap
from typing import Any
from urllib.parse import parse_qs, urlparse

import requests
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)


# ── Video ID extraction ───────────────────────────────────────────────────────

def extract_video_id(url: str) -> str | None:
    """
    Parse a YouTube video ID from a variety of URL formats:
      - https://www.youtube.com/watch?v=VIDEO_ID
      - https://youtu.be/VIDEO_ID
      - https://youtube.com/shorts/VIDEO_ID
      - https://www.youtube.com/embed/VIDEO_ID
    Returns None if no valid ID is found.
    """
    url = url.strip()
    parsed = urlparse(url)

    # youtu.be short links
    if parsed.netloc in ("youtu.be", "www.youtu.be"):
        vid = parsed.path.lstrip("/").split("/")[0]
        return vid if _valid_id(vid) else None

    # Standard youtube.com paths
    if "youtube.com" in parsed.netloc:
        # /watch?v=...
        qs = parse_qs(parsed.query)
        if "v" in qs:
            vid = qs["v"][0]
            return vid if _valid_id(vid) else None
        # /embed/ or /shorts/
        match = re.search(r"(?:embed|shorts|v)/([A-Za-z0-9_-]{11})", parsed.path)
        if match:
            return match.group(1)

    # Last-resort bare ID (11 char alphanumeric + - _)
    match = re.fullmatch(r"[A-Za-z0-9_-]{11}", url)
    if match:
        return url

    return None


def _valid_id(vid: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9_-]{11}", vid))


# ── Video metadata (oembed, no API key needed) ────────────────────────────────

def fetch_video_metadata(video_id: str) -> dict[str, Any] | None:
    """
    Fetch lightweight metadata via YouTube oEmbed.
    Returns a dict with keys: title, author, thumbnail, duration, views.
    duration and views are best-effort (not available from oEmbed).
    """
    try:
        oembed_url = (
            f"https://www.youtube.com/oembed"
            f"?url=https://www.youtube.com/watch?v={video_id}&format=json"
        )
        resp = requests.get(oembed_url, timeout=8)
        if resp.status_code == 200:
            data = resp.json()
            return {
                "title":     data.get("title", "Unknown Title"),
                "author":    data.get("author_name", "Unknown"),
                "thumbnail": data.get("thumbnail_url", ""),
                "duration":  "N/A",   # oEmbed doesn't expose duration
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
    Fetch a YouTube transcript using youtube-transcript-api.

    Returns:
        (transcript_text, error_message)
        On success error_message is None.
    """
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        # Prefer manually created English; fallback to auto-generated; then any
        try:
            transcript = transcript_list.find_manually_created_transcript(["en"])
        except NoTranscriptFound:
            try:
                transcript = transcript_list.find_generated_transcript(["en"])
            except NoTranscriptFound:
                transcript = next(iter(transcript_list))

        segments = transcript.fetch()

        if include_timestamps:
            lines = []
            for seg in segments:
                ts  = _format_timestamp(seg.start)
                lines.append(f"[{ts}] {seg.text}")
            return "\n".join(lines), None
        else:
            return " ".join(seg.text for seg in segments), None

    except TranscriptsDisabled:
        return "", "Transcripts are disabled for this video."
    except VideoUnavailable:
        return "", "Video is unavailable or private."
    except NoTranscriptFound:
        return "", "No transcript found (video may not have captions)."
    except Exception as exc:
        return "", f"Unexpected error: {exc}"


def _format_timestamp(seconds: float) -> str:
    seconds = int(seconds)
    h, r = divmod(seconds, 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


# ── Gemini API helper ─────────────────────────────────────────────────────────

_GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.0-flash:generateContent"
)

_STYLE_INSTRUCTIONS: dict[str, str] = {
    "Concise":         "Provide a concise summary in 3-5 short paragraphs.",
    "Detailed":        "Provide a thorough, detailed summary covering all major topics discussed.",
    "Bullet Points":   "Summarize using clearly organized bullet points grouped by topic.",
    "Executive Brief": (
        "Write an executive brief: one-line TL;DR, followed by 3-5 high-impact insights, "
        "and a recommended action or takeaway."
    ),
}


def _call_gemini(prompt: str, api_key: str, max_tokens: int = 1024) -> str:
    """Low-level call to Gemini generateContent REST endpoint."""
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": 0.4,
            "topP": 0.9,
        },
    }
    resp = requests.post(
        f"{_GEMINI_ENDPOINT}?key={api_key}",
        headers=headers,
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError) as exc:
        raise ValueError(f"Unexpected Gemini response structure: {data}") from exc


# ── Public AI functions ───────────────────────────────────────────────────────

def summarize_with_gemini(
    transcript: str,
    api_key: str,
    style: str = "Concise",
    language: str = "English",
    max_tokens: int = 1024,
) -> str:
    """Generate a summary of the transcript using Gemini."""
    style_instruction = _STYLE_INSTRUCTIONS.get(style, _STYLE_INSTRUCTIONS["Concise"])
    # Truncate transcript to stay within model context limits (~30k chars)
    safe_transcript = transcript[:30_000]

    prompt = textwrap.dedent(f"""
        You are an expert content analyst. Your task is to summarize a YouTube video transcript.

        Style instruction: {style_instruction}
        Output language: {language}

        Transcript:
        {safe_transcript}

        Provide only the summary. Do not include preamble or meta-commentary.
    """).strip()

    return _call_gemini(prompt, api_key, max_tokens)


def generate_key_points(
    transcript: str,
    api_key: str,
    max_tokens: int = 1024,
) -> list[str] | str:
    """
    Extract the top key points from the transcript.
    Returns a list of strings if JSON parsing succeeds, else raw text.
    """
    safe_transcript = transcript[:30_000]
    prompt = textwrap.dedent(f"""
        Extract the 7 most important key points from the following YouTube transcript.
        Respond ONLY with a valid JSON array of strings, e.g.:
        ["Point one.", "Point two.", ...]
        Do not include any other text.

        Transcript:
        {safe_transcript}
    """).strip()

    raw = _call_gemini(prompt, api_key, max_tokens)

    # Try to parse JSON
    try:
        # Strip possible markdown fences
        clean = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
        points = json.loads(clean)
        if isinstance(points, list):
            return [str(p) for p in points]
    except (json.JSONDecodeError, ValueError):
        pass

    return raw  # fallback: return raw text


def generate_quiz(
    transcript: str,
    api_key: str,
    n_questions: int = 5,
) -> list[dict[str, str]] | str:
    """
    Generate n quiz Q&A pairs from the transcript.
    Returns list of {question, answer, explanation} dicts on success.
    """
    safe_transcript = transcript[:25_000]
    prompt = textwrap.dedent(f"""
        Based on the following YouTube transcript, create {n_questions} comprehension questions.
        Respond ONLY with a valid JSON array like:
        [
          {{
            "question": "...",
            "answer": "...",
            "explanation": "..."
          }}
        ]
        Do not include any other text or markdown fences.

        Transcript:
        {safe_transcript}
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


# ── Export helpers ────────────────────────────────────────────────────────────

def export_summary_as_txt(
    title: str,
    summary: str,
    key_points: list[str] | str,
) -> str:
    """Return a plain-text export string of the summary + key points."""
    sep = "=" * 60
    lines = [
        sep,
        f"YouTube Video Summary: {title}",
        sep,
        "",
        "SUMMARY",
        "-------",
        summary,
        "",
        "KEY POINTS",
        "----------",
    ]
    if isinstance(key_points, list):
        for i, pt in enumerate(key_points, 1):
            lines.append(f"{i}. {pt}")
    else:
        lines.append(key_points)
    lines += ["", sep]
    return "\n".join(lines)
