"""
utils.py — Core logic for YouTube Video Summarizer & Q&A Bot
=============================================================
Handles:
  - YouTube video ID extraction
  - Transcript fetching (with timestamps, multilingual)
  - Text chunking with overlap
  - FAISS vector store creation (Google Embeddings)
  - RAG-based Q&A with Gemini 2.0 Flash
  - Full video summarization with timestamps
"""

import re
from typing import Optional

from youtube_transcript_api import YouTubeTranscriptApi
try:
    from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
except ImportError:
    from youtube_transcript_api import TranscriptsDisabled, NoTranscriptFound

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

import google.generativeai as genai


# ── 1. Extract YouTube video ID ────────────────────────────────────────────────

def extract_video_id(url: str) -> Optional[str]:
    patterns = [
        r"(?:v=)([A-Za-z0-9_-]{11})",
        r"youtu\.be/([A-Za-z0-9_-]{11})",
        r"embed/([A-Za-z0-9_-]{11})",
        r"shorts/([A-Za-z0-9_-]{11})",
        r"m\.youtube\.com/watch\?v=([A-Za-z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


# ── 2. Fetch transcript with timestamps (multilingual) ────────────────────────

def fetch_transcript(video_id: str, preferred_lang: str = "en") -> list[Document]:
    """
    Fetch transcript - compatible with all youtube-transcript-api versions.
    """
    raw_entries = None
    last_error = None

    # Strategy 1: get_transcript() directly (works in most versions)
    try:
        raw_entries = YouTubeTranscriptApi.get_transcript(video_id, languages=[preferred_lang])
    except Exception as e:
        last_error = e

    # Strategy 2: get_transcript() without language preference
    if raw_entries is None:
        try:
            raw_entries = YouTubeTranscriptApi.get_transcript(video_id)
        except Exception as e:
            last_error = e

    # Strategy 3: list_transcripts() then fetch (newer API)
    if raw_entries is None:
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            transcript = None
            try:
                transcript = transcript_list.find_transcript([preferred_lang])
            except Exception:
                pass
            if transcript is None:
                try:
                    transcript = transcript_list.find_generated_transcript([preferred_lang])
                except Exception:
                    pass
            if transcript is None:
                transcript = list(transcript_list)[0]

            fetched = transcript.fetch()

            # New API returns FetchedTranscript object — iterate it
            entries_temp = []
            for item in fetched:
                if hasattr(item, 'text') and hasattr(item, 'start'):
                    entries_temp.append({"text": item.text, "start": item.start})
                elif isinstance(item, dict):
                    entries_temp.append(item)
            raw_entries = entries_temp if entries_temp else None
        except Exception as e:
            last_error = e

    if not raw_entries:
        raise ValueError(f"Could not fetch transcript. The video may have captions disabled, or be private/age-restricted. Detail: {last_error}")

    # Normalize entries — handle both dict and object formats
    entries = []
    for item in raw_entries:
        if isinstance(item, dict):
            entries.append({"text": item.get("text", ""), "start": float(item.get("start", 0))})
        elif hasattr(item, 'text'):
            entries.append({"text": item.text, "start": float(item.start)})

    if not entries:
        raise ValueError("Transcript is empty.")

    WORDS_PER_CHUNK = 300
    OVERLAP_ENTRIES = 3

    chunks: list[Document] = []
    current_text = []
    current_start = entries[0]["start"]
    word_count = 0
    i = 0

    while i < len(entries):
        entry = entries[i]
        words = entry["text"].split()
        current_text.append(entry["text"])
        word_count += len(words)

        if word_count >= WORDS_PER_CHUNK or i == len(entries) - 1:
            chunk_text = " ".join(current_text).strip()
            chunks.append(Document(
                page_content=chunk_text,
                metadata={
                    "start": int(current_start),
                    "video_id": video_id,
                },
            ))
            overlap_start = max(0, i - OVERLAP_ENTRIES + 1)
            i = overlap_start
            current_start = entries[i]["start"] if i < len(entries) else current_start
            current_text = []
            word_count = 0

        i += 1

    return chunks


def get_available_languages(video_id: str) -> list[dict]:
    """Return list of available transcript languages for a video."""
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        langs = []
        for t in transcript_list:
            langs.append({
                "code": t.language_code,
                "name": t.language,
                "auto": t.is_generated,
            })
        return langs
    except Exception:
        return []


# ── 3. Build FAISS vector store with Google Embeddings ───────────────────────

def build_vector_store(chunks: list[Document], gemini_key: str) -> FAISS:
    """Embed transcript chunks with Google embeddings and store in FAISS."""
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=gemini_key,
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


# ── 4. Full Video Summarization with Timestamps ───────────────────────────────

def summarize_video(
    chunks: list[Document],
    gemini_key: str,
    output_language: str = "English",
    summary_style: str = "Detailed",
) -> str:
    """
    Generate a full video summary with timestamped sections using Gemini 2.0 Flash.
    Supports output in any language.
    """
    genai.configure(api_key=gemini_key)

    transcript_text = ""
    for doc in chunks:
        ts = format_timestamp(doc.metadata.get("start", 0))
        transcript_text += f"[{ts}] {doc.page_content}\n\n"

    style_instructions = {
        "Brief": "Write a concise 3-5 sentence overview only.",
        "Detailed": "Write a detailed summary with key points, organized by topic sections.",
        "Bullet Points": "Summarize using bullet points grouped under topic headings.",
        "Chapter-by-Chapter": "Break the video into chapters with a title and summary for each section.",
    }

    prompt = f"""You are an expert video summarizer.

Analyze the following YouTube video transcript and produce a high-quality summary.

OUTPUT LANGUAGE: {output_language}
SUMMARY STYLE: {style_instructions.get(summary_style, style_instructions["Detailed"])}

REQUIREMENTS:
1. Start with a 2-sentence TL;DR overview.
2. Include timestamped sections like [MM:SS] Topic Title — for every major topic shift.
3. Highlight key insights, facts, or takeaways using ✦ bullets.
4. End with a "Key Takeaways" section listing the 3-5 most important points.
5. Write everything in {output_language}.

TRANSCRIPT:
{transcript_text[:14000]}

Produce the summary now:"""

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text


# ── 5. RAG Q&A with Gemini 2.0 Flash ─────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert video assistant that answers questions strictly based on a YouTube video transcript.

Rules:
- Only use information from the transcript context provided.
- If the answer is not in the transcript, say: "I couldn't find that in the video."
- Be concise and direct. Use bullet points for lists.
- When referencing specific moments, mention the timestamp.
- Never make up information or draw on outside knowledge.
- Respond in the same language as the user's question.

Transcript context:
{context}
"""

def get_answer(
    question: str,
    vector_store: FAISS,
    gemini_key: str,
    chat_history: list[dict] = None,
    top_k: int = 5,
) -> tuple[str, list[int]]:
    """
    Retrieve relevant transcript chunks and generate an answer using Gemini 2.0 Flash.
    Returns answer string and list of source timestamps (seconds).
    """
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )
    relevant_docs = retriever.invoke(question)
    relevant_docs.sort(key=lambda d: d.metadata.get("start", 0))

    context_parts = []
    for doc in relevant_docs:
        ts = format_timestamp(doc.metadata.get("start", 0))
        context_parts.append(f"[{ts}] {doc.page_content}")
    context = "\n\n".join(context_parts)

    messages = [SystemMessage(content=SYSTEM_PROMPT.format(context=context))]

    if chat_history:
        for msg in chat_history[-8:]:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))

    messages.append(HumanMessage(content=question))

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=gemini_key,
        temperature=0.2,
        max_output_tokens=1024,
    )
    response = llm.invoke(messages)
    answer = response.content

    sources = sorted({
        int(doc.metadata.get("start", 0))
        for doc in relevant_docs
        if doc.metadata.get("start") is not None
    })

    return answer, sources


# ── 6. Timestamp formatter ────────────────────────────────────────────────────

def format_timestamp(seconds: int) -> str:
    """Convert seconds to MM:SS or H:MM:SS format."""
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"
