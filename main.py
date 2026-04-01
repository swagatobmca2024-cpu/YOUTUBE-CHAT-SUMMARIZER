"""
main.py — YouTube Video Summarizer & Q&A Bot
=============================================
Powered by Gemini 2.0 Flash · Dark Themed · Multilingual

Features:
  - Full video summarization with timestamps
  - RAG-based Q&A chat
  - Multilingual output support
  - Multiple summary styles
  - Clickable timestamp links
  - Chat history with export

Run:
    streamlit run main.py
"""

import streamlit as st
from utils import (
    extract_video_id,
    fetch_transcript,
    build_vector_store,
    get_answer,
    summarize_video,
    get_available_languages,
    format_timestamp,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="YT Summarizer & Q&A",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark Theme CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500;600&display=swap');

    /* Global dark background */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
        background-color: #0d0d0d !important;
        color: #e8e8e8 !important;
        font-family: 'Inter', sans-serif;
    }

    [data-testid="stSidebar"] {
        background-color: #111111 !important;
        border-right: 1px solid #222 !important;
    }

    [data-testid="stSidebar"] * {
        color: #e8e8e8 !important;
    }

    /* Header */
    .yt-header {
        background: linear-gradient(135deg, #1a0a0a 0%, #0d0d0d 50%, #0a0a1a 100%);
        border: 1px solid #2a2a2a;
        border-radius: 16px;
        padding: 28px 32px;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
    }
    .yt-header::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, #ff0000, #ff4444, #ff0000);
    }
    .yt-header h1 {
        font-family: 'Space Mono', monospace;
        font-size: 1.8rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0 0 6px 0;
        letter-spacing: -0.5px;
    }
    .yt-header p {
        color: #888;
        font-size: 0.88rem;
        margin: 0;
        font-weight: 300;
    }
    .yt-badge {
        display: inline-block;
        background: #ff0000;
        color: white;
        font-size: 10px;
        font-weight: 700;
        font-family: 'Space Mono', monospace;
        padding: 3px 8px;
        border-radius: 4px;
        letter-spacing: 1px;
        margin-right: 8px;
        vertical-align: middle;
    }
    .gemini-badge {
        display: inline-block;
        background: linear-gradient(90deg, #4285f4, #34a853);
        color: white;
        font-size: 10px;
        font-weight: 700;
        font-family: 'Space Mono', monospace;
        padding: 3px 8px;
        border-radius: 4px;
        letter-spacing: 0.5px;
        vertical-align: middle;
    }

    /* Video meta card */
    .video-meta {
        background: #141414;
        border: 1px solid #222;
        border-left: 3px solid #ff0000;
        border-radius: 10px;
        padding: 14px 18px;
        font-size: 13px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .video-meta a {
        color: #aaa;
        text-decoration: none;
    }
    .video-meta a:hover { color: #ff4444; }
    .chunk-count {
        background: #1e1e1e;
        color: #666;
        font-size: 11px;
        padding: 3px 10px;
        border-radius: 20px;
        font-family: 'Space Mono', monospace;
        margin-left: auto;
    }

    /* Tabs styling */
    [data-testid="stTabs"] button {
        font-family: 'Space Mono', monospace !important;
        font-size: 13px !important;
        color: #666 !important;
        background: transparent !important;
        border: none !important;
        padding: 10px 20px !important;
    }
    [data-testid="stTabs"] button[aria-selected="true"] {
        color: #ffffff !important;
        border-bottom: 2px solid #ff0000 !important;
    }

    /* Summary output box */
    .summary-box {
        background: #111;
        border: 1px solid #222;
        border-radius: 12px;
        padding: 24px 28px;
        font-size: 14px;
        line-height: 1.8;
        color: #ddd;
        white-space: pre-wrap;
    }

    /* Timestamp badge */
    .ts-badge {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        background: #1a1a1a;
        border: 1px solid #333;
        color: #ff4444;
        font-size: 11px;
        padding: 4px 10px;
        border-radius: 20px;
        margin: 3px 3px 0 0;
        font-family: 'Space Mono', monospace;
        text-decoration: none;
        transition: all 0.2s;
    }
    .ts-badge:hover {
        background: #ff0000;
        color: white;
        border-color: #ff0000;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        background: #111 !important;
        border: 1px solid #1e1e1e !important;
        border-radius: 12px !important;
        margin-bottom: 8px !important;
    }

    /* Chat input */
    [data-testid="stChatInputTextArea"] {
        background: #141414 !important;
        border: 1px solid #2a2a2a !important;
        color: #e8e8e8 !important;
        border-radius: 10px !important;
    }

    /* Input fields */
    [data-testid="stTextInput"] input,
    [data-testid="stSelectbox"] select {
        background: #141414 !important;
        border: 1px solid #2a2a2a !important;
        color: #e8e8e8 !important;
        border-radius: 8px !important;
    }

    /* Buttons */
    [data-testid="stButton"] button[kind="primary"] {
        background: #ff0000 !important;
        border: none !important;
        color: white !important;
        font-family: 'Space Mono', monospace !important;
        font-size: 13px !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        letter-spacing: 0.5px !important;
    }
    [data-testid="stButton"] button[kind="primary"]:hover {
        background: #cc0000 !important;
    }
    [data-testid="stButton"] button[kind="secondary"] {
        background: #1a1a1a !important;
        border: 1px solid #333 !important;
        color: #aaa !important;
        border-radius: 8px !important;
    }

    /* Info / success / error boxes */
    [data-testid="stAlert"] {
        background: #141414 !important;
        border-radius: 8px !important;
        border: 1px solid #2a2a2a !important;
    }

    /* Selectbox */
    [data-testid="stSelectbox"] > div > div {
        background: #141414 !important;
        border: 1px solid #2a2a2a !important;
        color: #e8e8e8 !important;
    }

    /* Sidebar sections */
    .sidebar-section {
        background: #1a1a1a;
        border: 1px solid #222;
        border-radius: 10px;
        padding: 14px;
        margin-bottom: 14px;
    }
    .sidebar-label {
        font-family: 'Space Mono', monospace;
        font-size: 10px;
        color: #555;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 10px;
    }

    /* Stats row */
    .stat-card {
        background: #141414;
        border: 1px solid #1e1e1e;
        border-radius: 8px;
        padding: 12px 16px;
        text-align: center;
    }
    .stat-value {
        font-family: 'Space Mono', monospace;
        font-size: 1.4rem;
        font-weight: 700;
        color: #ff4444;
    }
    .stat-label {
        font-size: 11px;
        color: #555;
        margin-top: 2px;
    }

    /* Divider */
    hr { border-color: #1e1e1e !important; }

    /* Hide Streamlit default elements */
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)

# ── Session state init ─────────────────────────────────────────────────────────
defaults = {
    "messages": [],
    "vector_store": None,
    "video_id": None,
    "transcript_chunks": [],
    "video_loaded": False,
    "summary": None,
    "gemini_key": "",
    "available_langs": [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="yt-header">
    <h1>
        <span class="yt-badge">YT</span>
        Summarizer & Q&amp;A
        <span class="gemini-badge">Gemini 2.0 Flash</span>
    </h1>
    <p>Paste any YouTube link · Get a timestamped summary · Chat with the video · Any language</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-label">🔑 API Configuration</div>', unsafe_allow_html=True)
    gemini_key = st.text_input(
        "Gemini API Key",
        type="password",
        placeholder="AIzaSy...",
        help="Get your free key at aistudio.google.com",
        value=st.session_state.gemini_key,
    )
    if gemini_key:
        st.session_state.gemini_key = gemini_key

    st.divider()
    st.markdown('<div class="sidebar-label">🎬 Video Input</div>', unsafe_allow_html=True)

    youtube_url = st.text_input(
        "YouTube URL",
        placeholder="https://www.youtube.com/watch?v=...",
    )

    st.markdown('<div class="sidebar-label">⚙️ Summary Options</div>', unsafe_allow_html=True)

    summary_style = st.selectbox(
        "Summary Style",
        ["Detailed", "Brief", "Bullet Points", "Chapter-by-Chapter"],
        index=0,
    )

    output_language = st.selectbox(
        "Output Language",
        [
            "English", "Hindi", "Bengali", "Spanish", "French",
            "German", "Portuguese", "Arabic", "Japanese", "Chinese",
            "Korean", "Italian", "Russian", "Turkish", "Dutch",
        ],
        index=0,
        help="Summary and answers will be generated in this language",
    )

    load_btn = st.button("🚀 Load & Summarize", type="primary", use_container_width=True)

    if st.session_state.video_loaded:
        st.divider()
        if st.button("🔄 Load New Video", use_container_width=True):
            for k, v in defaults.items():
                st.session_state[k] = v
            st.rerun()

        # Stats
        chunks = st.session_state.transcript_chunks
        if chunks:
            st.divider()
            st.markdown('<div class="sidebar-label">📊 Video Stats</div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-value">{len(chunks)}</div>
                    <div class="stat-label">Chunks</div>
                </div>""", unsafe_allow_html=True)
            with col2:
                total_words = sum(len(c.page_content.split()) for c in chunks)
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-value">{total_words // 1000}K</div>
                    <div class="stat-label">Words</div>
                </div>""", unsafe_allow_html=True)

            # Duration estimate
            last_ts = chunks[-1].metadata.get("start", 0)
            st.markdown(f"""
            <div class="stat-card" style="margin-top:10px">
                <div class="stat-value">{format_timestamp(last_ts)}</div>
                <div class="stat-label">~Duration</div>
            </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="sidebar-label">ℹ️ How it works</div>', unsafe_allow_html=True)
    st.caption("1. Fetches transcript from YouTube")
    st.caption("2. Chunks text with timestamps")
    st.caption("3. Embeds with Google AI")
    st.caption("4. Summarizes with Gemini 2.0 Flash")
    st.caption("5. Answers questions via RAG")

# ── Load Video ─────────────────────────────────────────────────────────────────
if load_btn:
    if not st.session_state.gemini_key:
        st.error("⚠️ Please enter your Gemini API Key in the sidebar.")
    elif not youtube_url:
        st.error("⚠️ Please paste a YouTube URL.")
    else:
        video_id = extract_video_id(youtube_url)
        if not video_id:
            st.error("❌ Could not parse a video ID. Make sure it's a valid YouTube link.")
        else:
            with st.spinner("Fetching transcript, building index & generating summary…"):
                try:
                    chunks = fetch_transcript(video_id)
                    if not chunks:
                        st.error("No transcript found. Captions may be disabled.")
                    else:
                        vs = build_vector_store(chunks, st.session_state.gemini_key)
                        summary = summarize_video(
                            chunks,
                            st.session_state.gemini_key,
                            output_language=output_language,
                            summary_style=summary_style,
                        )
                        langs = get_available_languages(video_id)

                        st.session_state.vector_store = vs
                        st.session_state.video_id = video_id
                        st.session_state.transcript_chunks = chunks
                        st.session_state.video_loaded = True
                        st.session_state.messages = []
                        st.session_state.summary = summary
                        st.session_state.available_langs = langs
                        st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {e}")

# ── Main Content ───────────────────────────────────────────────────────────────
if not st.session_state.video_loaded:
    st.info("👈 Enter your Gemini API key and a YouTube URL in the sidebar, then click **Load & Summarize**.")

    # Feature cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="stat-card" style="padding:20px;text-align:left">
            <div style="font-size:24px;margin-bottom:8px">📝</div>
            <div style="font-weight:600;margin-bottom:4px;color:#fff">Smart Summarization</div>
            <div style="font-size:12px;color:#555">Timestamped summaries in 4 styles — brief, detailed, bullets, or chapters</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="stat-card" style="padding:20px;text-align:left">
            <div style="font-size:24px;margin-bottom:8px">💬</div>
            <div style="font-weight:600;margin-bottom:4px;color:#fff">Chat with Video</div>
            <div style="font-size:12px;color:#555">Ask anything about the video — get answers with clickable timestamp links</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="stat-card" style="padding:20px;text-align:left">
            <div style="font-size:24px;margin-bottom:8px">🌍</div>
            <div style="font-weight:600;margin-bottom:4px;color:#fff">Any Language</div>
            <div style="font-size:12px;color:#555">Get summaries and answers in 15+ languages including Hindi & Bengali</div>
        </div>""", unsafe_allow_html=True)

else:
    vid = st.session_state.video_id

    # Video meta bar
    st.markdown(f"""
    <div class="video-meta">
        <span style="font-size:20px">▶️</span>
        <a href="https://www.youtube.com/watch?v={vid}" target="_blank">
            youtube.com/watch?v={vid}
        </a>
        <span class="chunk-count">{len(st.session_state.transcript_chunks)} chunks · {output_language}</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Tabs ───────────────────────────────────────────────────────────────────
    tab1, tab2 = st.tabs(["📝 Summary", "💬 Chat with Video"])

    # ── Tab 1: Summary ─────────────────────────────────────────────────────────
    with tab1:
        if st.session_state.summary:
            # Action buttons row
            col1, col2, col3 = st.columns([1, 1, 4])
            with col1:
                if st.button("🔄 Regenerate", use_container_width=True):
                    with st.spinner("Regenerating summary…"):
                        try:
                            new_summary = summarize_video(
                                st.session_state.transcript_chunks,
                                st.session_state.gemini_key,
                                output_language=output_language,
                                summary_style=summary_style,
                            )
                            st.session_state.summary = new_summary
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")
            with col2:
                st.download_button(
                    "💾 Export",
                    data=st.session_state.summary,
                    file_name=f"summary_{vid}.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

            st.markdown("---")
            st.markdown(st.session_state.summary)

        else:
            st.info("Summary will appear here after loading a video.")

    # ── Tab 2: Q&A Chat ────────────────────────────────────────────────────────
    with tab2:
        # Available languages info
        if st.session_state.available_langs:
            lang_names = [f"{l['name']} {'(auto)' if l['auto'] else ''}" for l in st.session_state.available_langs[:5]]
            st.caption(f"📡 Transcript languages available: {', '.join(lang_names)}")

        st.caption("💡 Ask anything about the video. Answers include clickable timestamp links.")

        # Render existing messages
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("sources"):
                    st.markdown("**Jump to:**")
                    badges = ""
                    for src in msg["sources"][:5]:
                        ts = format_timestamp(src)
                        url = f"https://www.youtube.com/watch?v={vid}&t={src}s"
                        badges += f'<a href="{url}" target="_blank" class="ts-badge">▶ {ts}</a>'
                    st.markdown(badges, unsafe_allow_html=True)

        # Chat input
        if prompt := st.chat_input("Ask anything about this video…"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    try:
                        answer, sources = get_answer(
                            question=prompt,
                            vector_store=st.session_state.vector_store,
                            gemini_key=st.session_state.gemini_key,
                            chat_history=st.session_state.messages[:-1],
                        )
                        st.markdown(answer)

                        if sources:
                            st.markdown("**Jump to:**")
                            badges = ""
                            for src in sources[:5]:
                                ts = format_timestamp(src)
                                url = f"https://www.youtube.com/watch?v={vid}&t={src}s"
                                badges += f'<a href="{url}" target="_blank" class="ts-badge">▶ {ts}</a>'
                            st.markdown(badges, unsafe_allow_html=True)

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                        })

                    except Exception as e:
                        err = f"❌ Error: {e}"
                        st.error(err)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": err,
                            "sources": [],
                        })

        # Export chat
        if st.session_state.messages:
            st.divider()
            chat_export = "\n\n".join(
                f"{'USER' if m['role'] == 'user' else 'ASSISTANT'}: {m['content']}"
                for m in st.session_state.messages
            )
            st.download_button(
                "💾 Export Chat",
                data=chat_export,
                file_name=f"chat_{vid}.txt",
                mime="text/plain",
            )
