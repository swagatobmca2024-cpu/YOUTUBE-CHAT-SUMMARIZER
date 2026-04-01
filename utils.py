"""
utils.py — Core logic for YouTube Q&A Bot (Gemini Edition)
===========================================================
Handles:
  - YouTube video ID extraction
  - Transcript fetching with timestamps
  - Text chunking with overlap
  - FAISS vector store with Google Embeddings
  - RAG-based Q&A with Gemini 2.0 Flash
"""

import re
from typing import Optional

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.schema import Document, HumanMessage, AIMessage, SystemMessage


# ── 1. Extract YouTube Video ID ───────────────────────────────────────────────

def extract_video_id(url: str) -> Optional[str]:
    """
    Parse a YouTube video ID from any common URL format.
    Supports: watch?v=, youtu.be/, /embed/, /shorts/
    """
    patterns = [
        r"(?:v=)([A-Za-z0-9_-]{11})",
        r"youtu\.be/([A-Za-z0-9_-]{11})",
        r"embed/([A-Za-z0-9_-]{11})",
        r"shorts/([A-Za-z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


# ── 2. Fetch Transcript with Timestamps ──────────────────────────────────────

def fetch_transcript(video_id: str) -> list[Document]:
    """
    Fetch transcript for a YouTube video. Returns LangChain Documents
    with start-time metadata. Tries English first, falls back to any available.
    """
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        # Priority: manual English → auto English → any manual → any auto
        try:
            transcript = transcript_list.find_manually_created_transcript(["en"])
        except Exception:
            try:
                transcript = transcript_list.find_generated_transcript(["en"])
            except Exception:
                # Grab whatever is available
                all_transcripts = list(transcript_list)
                if not all_transcripts:
                    raise ValueError("No transcripts found for this video.")
                transcript = all_transcripts[0]

        entries = transcript.fetch()

    except TranscriptsDisabled:
        raise ValueError("Transcripts are disabled for this video.")
    except NoTranscriptFound:
        raise ValueError("No transcript found. The video may not have captions.")
    except Exception as e:
        raise ValueError(f"Failed to fetch transcript: {e}")

    if not entries:
        raise ValueError("Transcript is empty.")

    # ── Chunk entries into overlapping segments ────────────────────────────────
    WORDS_PER_CHUNK = 300
    OVERLAP_ENTRIES = 3

    chunks: list[Document] = []
    current_text: list[str] = []
    current_start: float = entries[0]["start"]
    word_count = 0
    i = 0

    while i < len(entries):
        entry = entries[i]
        words = entry["text"].split()
        current_text.append(entry["text"])
        word_count += len(words)

        if word_count >= WORDS_PER_CHUNK or i == len(entries) - 1:
            chunk_text = " ".join(current_text).strip()
            if chunk_text:
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


# ── 3. Build FAISS Vector Store with Google Embeddings ───────────────────────

def build_vector_store(chunks: list[Document], gemini_key: str) -> FAISS:
    """
    Embed transcript chunks with Google's embedding model
    and store in an in-memory FAISS index.
    Uses: models/embedding-001 (free tier, 768-dim)
    """
    genai.configure(api_key=gemini_key)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=gemini_key,
    )

    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


# ── 4. RAG Q&A with Gemini 2.0 Flash ─────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert video analyst assistant. Your job is to answer questions about a YouTube video using ONLY the transcript excerpts provided below.

STRICT RULES:
- Base your answer ENTIRELY on the transcript context below.
- If the answer is not in the transcript, respond: "I couldn't find that in this video."
- Be concise, clear, and direct.
- Use bullet points for multi-part answers.
- When quoting the video directly, use quotation marks.
- Never fabricate information or use outside knowledge.
- Reference timestamps like [2:34] when relevant.

TRANSCRIPT CONTEXT:
{context}
"""

def get_answer(
    question: str,
    vector_store: FAISS,
    gemini_key: str,
    model: str = "gemini-2.0-flash",
    chat_history: list[dict] = None,
    top_k: int = 6,
) -> tuple[str, list[int]]:
    """
    Retrieves relevant transcript chunks for the question,
    then generates an answer using Gemini 2.0 Flash.

    Returns:
        answer  : generated answer string
        sources : sorted list of start-time seconds
    """
    genai.configure(api_key=gemini_key)

    # Retrieve top-k relevant chunks via similarity search
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )
    relevant_docs = retriever.invoke(question)

    # Sort chronologically for readable context
    relevant_docs.sort(key=lambda d: d.metadata.get("start", 0))

    # Build context block
    context_parts = []
    for doc in relevant_docs:
        ts = format_timestamp(doc.metadata.get("start", 0))
        context_parts.append(f"[{ts}] {doc.page_content}")
    context = "\n\n".join(context_parts)

    # Construct message chain
    messages = [SystemMessage(content=SYSTEM_PROMPT.format(context=context))]

    if chat_history:
        for msg in chat_history[-8:]:  # keep last 8 turns for memory
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))

    messages.append(HumanMessage(content=question))

    # Call Gemini via LangChain
    llm = ChatGoogleGenerativeAI(
        model=model,
        google_api_key=gemini_key,
        temperature=0.2,
        max_output_tokens=1024,
    )
    response = llm.invoke(messages)
    answer = response.content

    # Extract unique sorted source timestamps
    sources = sorted({
        int(doc.metadata.get("start", 0))
        for doc in relevant_docs
        if doc.metadata.get("start") is not None
    })

    return answer, sources


# ── 5. Timestamp Formatter ────────────────────────────────────────────────────

def format_timestamp(seconds: int) -> str:
    """Convert seconds to MM:SS or H:MM:SS format."""
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


# ── 6. Video Duration Estimate ────────────────────────────────────────────────

def estimate_duration(chunks: list[Document]) -> str:
    """Estimate video duration from last chunk's timestamp."""
    if not chunks:
        return "Unknown"
    last_start = max(c.metadata.get("start", 0) for c in chunks)
    return format_timestamp(last_start)
