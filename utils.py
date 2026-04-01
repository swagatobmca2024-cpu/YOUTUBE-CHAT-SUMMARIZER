"""
utils.py — helper functions for YT Summarizer
Compatible with Python 3.11+  |  youtube-transcript-api v1.x
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
    CouldNotRetrieveTranscript,
    NoTranscriptFound,
    RequestBlocked,
    TranscriptsDisabled,
    VideoUnavailable,
)


# ── Browser-like headers (used when building sessions) ────────────────────────

_YT_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;"
        "q=0.9,image/webp,*/*;q=0.8"
    ),
    "Referer": "https://www.youtube.com/",
}


# ── Video ID extraction ───────────────────────────────────────────────────────

def extract_video_id(url: str) -> str | None:
    """
    Handles all YouTube URL formats including ?si= tracking params:
      youtu.be/ID?si=xxx  |  youtube.com/watch?v=ID&si=xxx
      youtube.com/shorts/ID  |  youtube.com/embed/ID
    """
    url = url.strip()
    parsed = urlparse(url)

    # youtu.be/VIDEO_ID
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

    # bare 11-char ID
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
    try:
        resp = requests.get(
            f"https://www.youtube.com/watch?v={video_id}",
            headers=_YT_HEADERS,
            timeout=12,
        )
        if resp.status_code != 200:
            return None
        html = resp.text

        # ── Title ─────────────────────────────────────────────────────────────
        title = "Unknown"
        og_title = re.search(r'<meta property="og:title"\s+content="([^"]+)"', html)
        if og_title:
            title = og_title.group(1)
        else:
            t2 = re.search(r'"title"\s*:\s*\{"runs":\[{"text":"([^"]+)"', html)
            if t2:
                title = t2.group(1)

        # ── Author ────────────────────────────────────────────────────────────
        author = "Unknown"
        a1 = re.search(r'"ownerChannelName"\s*:\s*"([^"]+)"', html)
        if a1:
            author = a1.group(1)
        else:
            a2 = re.search(r'"channelName"\s*:\s*"([^"]+)"', html)
            if a2:
                author = a2.group(1)
            else:
                a3 = re.search(r'"author"\s*:\s*"([^"]+)"', html)
                if a3:
                    author = a3.group(1)

        # ── Duration ──────────────────────────────────────────────────────────
        duration = "N/A"
        d1 = re.search(r'"lengthSeconds"\s*:\s*"(\d+)"', html)
        if d1:
            secs = int(d1.group(1))
            h, r = divmod(secs, 3600)
            mn, s = divmod(r, 60)
            duration = f"{h}:{mn:02d}:{s:02d}" if h else f"{mn}:{s:02d}"
        else:
            d2 = re.search(r'"duration"\s*:\s*"(PT[^"]+)"', html)
            if d2:
                duration = _parse_iso_duration(d2.group(1))

        # ── Views ─────────────────────────────────────────────────────────────
        views = "N/A"
        v1 = re.search(r'"viewCount"\s*:\s*"(\d+)"', html)
        if v1:
            views = _fmt_views(int(v1.group(1)))
        else:
            v2 = re.search(r'"videoViewCountRenderer".*?"simpleText"\s*:\s*"([^"]+)"', html)
            if v2:
                views = v2.group(1)

        # ── Thumbnail ─────────────────────────────────────────────────────────
        thumbnail = f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"
        th = re.search(r'<meta property="og:image"\s+content="([^"]+)"', html)
        if th:
            thumbnail = th.group(1)

        if title == "Unknown" and author == "Unknown":
            return None

        return {
            "title":     title,
            "author":    author,
            "thumbnail": thumbnail,
            "duration":  duration,
            "views":     views,
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
                "thumbnail": data.get(
                    "thumbnail_url",
                    f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg",
                ),
                "duration":  "N/A",
                "views":     "N/A",
            }
    except Exception:
        pass
    return None


# ── Cookie / session helpers ──────────────────────────────────────────────────

def _parse_netscape_cookies(cookies_txt: str) -> list[dict]:
    """
    Parse a Netscape-format cookies.txt into a list of cookie dicts.
    Each line (tab-separated):
      .youtube.com  TRUE  /  FALSE  0  COOKIE_NAME  cookie_value
    """
    cookies: list[dict] = []
    for line in cookies_txt.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 7:
            continue
        domain, _, path, secure, _, name, value = parts[:7]
        cookies.append(
            {
                "domain": domain,
                "path":   path,
                "secure": secure.upper() == "TRUE",
                "name":   name,
                "value":  value,
            }
        )
    return cookies


def _session_from_cookies(cookies_txt: str | None) -> requests.Session:
    """
    Build a requests.Session with browser-like headers + optional YT cookies.
    Providing cookies_txt (Netscape format) greatly improves success on
    cloud server IPs that YouTube otherwise blocks.
    """
    session = requests.Session()
    session.headers.update(_YT_HEADERS)

    # Basic consent cookies — always set
    session.cookies.set("CONSENT", "YES+cb", domain=".youtube.com")
    session.cookies.set(
        "SOCS",
        "CAESEwgDEgk0OTc5NTkzNzIaAmVuIAEaBgiAo_CmBg",
        domain=".youtube.com",
    )

    if cookies_txt:
        for ck in _parse_netscape_cookies(cookies_txt):
            session.cookies.set(
                ck["name"],
                ck["value"],
                domain=ck["domain"],
                path=ck["path"],
            )
    return session


# ── Transcript fetching ───────────────────────────────────────────────────────

def fetch_transcript(
    video_id: str,
    include_timestamps: bool = False,
    cookies_txt: str | None = None,
    proxy_user: str | None = None,
    proxy_pass: str | None = None,
) -> tuple[str, str | None]:
    """
    Fetch transcript using youtube-transcript-api v1.x.

    Attempt order (stops at first success):
      1. Webshare residential proxy  — bypasses Streamlit Cloud IP blocks
      2. Browser-spoofed session + cookies (if provided)
      3. Bare default session

    Returns (text, error_message). error_message is None on success.
    """
    from youtube_transcript_api.proxies import WebshareProxyConfig

    def _best_transcript(tlist):
        try:
            return tlist.find_manually_created_transcript(["en"])
        except NoTranscriptFound:
            pass
        try:
            return tlist.find_generated_transcript(["en"])
        except NoTranscriptFound:
            pass
        return next(iter(tlist))

    def _run(ytt: YouTubeTranscriptApi) -> tuple[str, str | None]:
        try:
            tlist = ytt.list(video_id)
            segs  = _best_transcript(tlist).fetch()
            return _segs_to_text(segs, include_timestamps), None
        except TranscriptsDisabled:
            return "", "Transcripts are disabled for this video."
        except VideoUnavailable:
            return "", "Video is unavailable or private."
        except NoTranscriptFound:
            return "", "No transcript found — video may not have captions."
        except (RequestBlocked, CouldNotRetrieveTranscript):
            return "", "__blocked__"
        except Exception as exc:
            return "", f"Unexpected error: {exc}"

    # ── Attempt 1: Webshare proxy (residential IP — works on Streamlit Cloud) ──
    if proxy_user and proxy_pass:
        try:
            proxy_cfg = WebshareProxyConfig(
                proxy_username=proxy_user,
                proxy_password=proxy_pass,
            )
            text, err = _run(YouTubeTranscriptApi(proxy_config=proxy_cfg))
            if text or (err and err != "__blocked__"):
                return text, err
        except Exception:
            pass  # proxy init failed — fall through to next attempt

    # ── Attempt 2: browser-spoofed session + optional cookies ─────────────────
    session = _session_from_cookies(cookies_txt)
    text, err = _run(YouTubeTranscriptApi(http_client=session))
    if text or (err and err != "__blocked__"):
        return text, err

    # ── Attempt 3: bare default session ───────────────────────────────────────
    text, err = _run(YouTubeTranscriptApi())
    if text or (err and err != "__blocked__"):
        return text, err

    # All attempts blocked
    if proxy_user and proxy_pass:
        return "", (
            "All fetch attempts failed (including Webshare proxy). "
            "Your proxy credentials may be incorrect or your free quota exhausted. "
            "Check your Webshare dashboard and verify the credentials in Streamlit secrets."
        )
    return "", (
        "YouTube is blocking transcript requests from this server's IP. "
        "**Fix:** add `WEBSHARE_USER` and `WEBSHARE_PASS` to your Streamlit secrets "
        "(free account at webshare.io is enough)."
    )


def _segs_to_text(segments: Any, include_timestamps: bool) -> str:
    if include_timestamps:
        return "\n".join(f"[{_fmt_ts(seg.start)}] {seg.text}" for seg in segments)
    return " ".join(seg.text for seg in segments)


def _fmt_ts(seconds: float) -> str:
    seconds = int(seconds)
    h, r    = divmod(seconds, 3600)
    m, s    = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


# ── Gemini helpers ────────────────────────────────────────────────────────────

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
    resp = requests.post(
        f"{_GEMINI_ENDPOINT}?key={api_key}",
        headers={"Content-Type": "application/json"},
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature":     0.4,
                "topP":            0.9,
            },
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError) as exc:
        raise ValueError(f"Unexpected Gemini response structure: {data}") from exc


def summarize_with_gemini(
    transcript: str,
    api_key: str,
    style: str = "Concise",
    language: str = "English",
    max_tokens: int = 1024,
) -> str:
    instr  = _STYLE_INSTRUCTIONS.get(style, _STYLE_INSTRUCTIONS["Concise"])
    prompt = textwrap.dedent(f"""
        You are an expert content analyst summarizing a YouTube video transcript.
        Style: {instr}
        Output language: {language}

        Transcript:
        {transcript[:30_000]}

        Provide only the summary with no preamble.
    """).strip()
    return _call_gemini(prompt, api_key, max_tokens)


def generate_key_points(
    transcript: str, api_key: str, max_tokens: int = 1024
) -> list[str] | str:
    prompt = textwrap.dedent(f"""
        Extract the 7 most important key points from this YouTube transcript.
        Respond ONLY with a valid JSON array of strings.
        Do not include any other text or markdown fences.

        Transcript:
        {transcript[:30_000]}
    """).strip()
    raw = _call_gemini(prompt, api_key, max_tokens)
    try:
        clean = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
        pts   = json.loads(clean)
        if isinstance(pts, list):
            return [str(p) for p in pts]
    except (json.JSONDecodeError, ValueError):
        pass
    return raw


def generate_quiz(
    transcript: str, api_key: str, n_questions: int = 5
) -> list[dict] | str:
    prompt = textwrap.dedent(f"""
        Create {n_questions} comprehension questions from this YouTube transcript.
        Respond ONLY with a valid JSON array (no markdown fences):
        [{{"question":"...","answer":"...","explanation":"..."}}]

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

def export_summary_as_txt(
    title: str, summary: str, key_points: list[str] | str
) -> str:
    sep   = "=" * 60
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
