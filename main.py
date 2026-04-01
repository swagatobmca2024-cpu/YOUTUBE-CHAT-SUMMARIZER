"""
main.py — YouTube Q&A Bot (Gemini Edition)
==========================================
Chat with any YouTube video. Powered by Gemini 2.0 Flash + Google Embeddings + FAISS.

Deploy:  streamlit run main.py
Cloud:   share.streamlit.io  →  set GEMINI_API_KEY in Secrets
"""

import streamlit as st
from utils import (
    extract_video_id,
    fetch_transcript,
    build_vector_store,
    get_answer,
    format_timestamp,
    estimate_duration,
)

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="YT Q&A · Gemini",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Advanced Dark Theme CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;700&family=Manrope:wght@300;400;600;800&display=swap');

/* ── Root Variables ── */
:root {
    --bg-base:       #080b10;
    --bg-surface:    #0d1117;
    --bg-elevated:   #161b22;
    --bg-hover:      #1c2130;
    --border:        #21262d;
    --border-accent: #30363d;
    --text-primary:  #e6edf3;
    --text-secondary:#8b949e;
    --text-muted:    #484f58;
    --accent-red:    #ff4d4d;
    --accent-amber:  #f0a500;
    --accent-teal:   #39d0d8;
    --accent-blue:   #58a6ff;
    --gem-from:      #4285f4;
    --gem-to:        #0f9d58;
    --radius-sm:     6px;
    --radius-md:     10px;
    --radius-lg:     16px;
}

/* ── Global Reset ── */
html, body, [class*="css"] {
    font-family: 'Manrope', sans-serif !important;
    background-color: var(--bg-base) !important;
    color: var(--text-primary) !important;
}

/* ── Hide Streamlit Defaults ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 4rem !important; max-width: 1100px; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--bg-surface) !important;
    border-right: 1px solid var(--border) !important;
    padding-top: 1rem;
}
section[data-testid="stSidebar"] .block-container {
    padding: 1rem !important;
}

/* ── Custom Header ── */
.app-header {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 0.25rem;
    padding-bottom: 1.25rem;
    border-bottom: 1px solid var(--border);
}
.app-logo {
    width: 42px;
    height: 42px;
    background: linear-gradient(135deg, var(--accent-red), #ff2d2d);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
    flex-shrink: 0;
    box-shadow: 0 0 20px rgba(255,77,77,0.3);
}
.app-title {
    font-family: 'Manrope', sans-serif;
    font-weight: 800;
    font-size: 1.55rem;
    color: var(--text-primary);
    line-height: 1.1;
}
.app-subtitle {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: var(--text-muted);
    margin-top: 2px;
    letter-spacing: 0.04em;
}
.gem-badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    font-weight: 600;
    color: var(--gem-from);
    background: rgba(66,133,244,0.1);
    border: 1px solid rgba(66,133,244,0.25);
    border-radius: 20px;
    padding: 2px 10px;
    display: inline-block;
    letter-spacing: 0.05em;
}

/* ── Sidebar Section Labels ── */
.sidebar-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: var(--text-muted);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin: 1.2rem 0 0.5rem 0;
    padding-bottom: 4px;
    border-bottom: 1px solid var(--border);
}

/* ── Video Meta Card ── */
.video-card {
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 14px 16px;
    margin-bottom: 1.25rem;
    display: flex;
    align-items: center;
    gap: 14px;
}
.video-thumb {
    width: 80px;
    height: 56px;
    border-radius: var(--radius-sm);
    object-fit: cover;
    flex-shrink: 0;
    border: 1px solid var(--border-accent);
}
.video-info {
    flex: 1;
    min-width: 0;
}
.video-url {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: var(--accent-blue);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.video-stats {
    display: flex;
    gap: 10px;
    margin-top: 6px;
    flex-wrap: wrap;
}
.stat-pill {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: var(--text-secondary);
    background: var(--bg-hover);
    border: 1px solid var(--border-accent);
    border-radius: 20px;
    padding: 2px 9px;
    display: inline-flex;
    align-items: center;
    gap: 4px;
}

/* ── Chat Messages ── */
.stChatMessage {
    background: transparent !important;
    border: none !important;
    padding: 0.3rem 0 !important;
}
[data-testid="stChatMessageContent"] {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    padding: 14px 18px !important;
    font-size: 0.95rem !important;
    line-height: 1.7 !important;
    color: var(--text-primary) !important;
}
[data-testid="stChatMessage"][data-role="user"] [data-testid="stChatMessageContent"] {
    background: var(--bg-hover) !important;
    border-color: var(--border-accent) !important;
}

/* ── Timestamp Badges ── */
.ts-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: var(--bg-elevated);
    border: 1px solid var(--border-accent);
    border-radius: 6px;
    padding: 4px 10px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: var(--accent-teal);
    text-decoration: none;
    transition: all 0.15s ease;
    margin: 3px;
}
.ts-badge:hover {
    background: rgba(57,208,216,0.1);
    border-color: var(--accent-teal);
    color: var(--accent-teal);
}
.ts-section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: var(--text-muted);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 10px;
    margin-bottom: 4px;
}

/* ── Input ── */
.stTextInput > div > div > input,
.stChatInputContainer textarea {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border-accent) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-primary) !important;
    font-family: 'Manrope', sans-serif !important;
    font-size: 0.9rem !important;
}
.stChatInputContainer {
    background: var(--bg-surface) !important;
    border-top: 1px solid var(--border) !important;
    padding-top: 0.75rem !important;
}

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border-accent) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
}

/* ── Primary Button ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #ff4d4d 0%, #cc2200 100%) !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    color: white !important;
    font-family: 'Manrope', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.02em !important;
    padding: 0.55rem 1.5rem !important;
    width: 100% !important;
    transition: opacity 0.2s, transform 0.1s !important;
    box-shadow: 0 0 18px rgba(255,77,77,0.2) !important;
}
.stButton > button[kind="primary"]:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}
.stButton > button[kind="secondary"] {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border-accent) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-secondary) !important;
    font-family: 'Manrope', sans-serif !important;
    font-size: 0.85rem !important;
    width: 100% !important;
}

/* ── Alert / Info ── */
.stAlert {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border-accent) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-secondary) !important;
}

/* ── Labels ── */
label, .stSelectbox label {
    color: var(--text-secondary) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.05em !important;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 0.75rem 0 !important; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--accent-teal) !important; }

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 3rem 2rem;
    color: var(--text-muted);
}
.empty-icon {
    font-size: 3rem;
    margin-bottom: 1rem;
    opacity: 0.5;
}
.empty-title {
    font-family: 'Manrope', sans-serif;
    font-weight: 600;
    font-size: 1.1rem;
    color: var(--text-secondary);
    margin-bottom: 0.5rem;
}
.empty-desc {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: var(--text-muted);
    line-height: 1.7;
}

/* ── How it works steps ── */
.how-step {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    margin-bottom: 10px;
    font-size: 0.82rem;
    color: var(--text-secondary);
    font-family: 'Manrope', sans-serif;
}
.step-num {
    width: 20px;
    height: 20px;
    background: var(--bg-hover);
    border: 1px solid var(--border-accent);
    border-radius: 50%;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    color: var(--accent-teal);
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    margin-top: 1px;
}
</style>
""", unsafe_allow_html=True)

# ── Session State Init ────────────────────────────────────────────────────────
defaults = {
    "messages": [],
    "vector_store": None,
    "video_id": None,
    "transcript_chunks": [],
    "video_loaded": False,
    "gemini_key": "",
    "model_choice": "gemini-2.0-flash",
    "chunk_count": 0,
    "video_duration": "—",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="app-header">
        <div class="app-logo">▶</div>
        <div>
            <div class="app-title">YT Q&amp;A Bot</div>
            <div class="app-subtitle">GEMINI · RAG · FAISS</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── API Key ──
    st.markdown('<div class="sidebar-label">🔑 API Key</div>', unsafe_allow_html=True)

    # Try to load from Streamlit secrets first (for cloud deployment)
    default_key = ""
    try:
        default_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        pass

    gemini_key = st.text_input(
        "Gemini API Key",
        type="password",
        value=default_key,
        placeholder="AIza...",
        help="Get yours free at aistudio.google.com",
        label_visibility="collapsed",
    )
    if gemini_key:
        st.session_state.gemini_key = gemini_key

    st.markdown(
        '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:0.68rem;'
        'color:#484f58;margin-top:4px;">Free at '
        '<a href="https://aistudio.google.com" target="_blank" style="color:#58a6ff;">aistudio.google.com</a>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Video Input ──
    st.markdown('<div class="sidebar-label">🎬 Video</div>', unsafe_allow_html=True)

    youtube_url = st.text_input(
        "YouTube URL",
        placeholder="https://youtube.com/watch?v=...",
        label_visibility="collapsed",
    )

    # ── Model ──
    st.markdown('<div class="sidebar-label">⚙️ Model</div>', unsafe_allow_html=True)
    model_choice = st.selectbox(
        "Gemini model",
        ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"],
        index=0,
        label_visibility="collapsed",
        help="gemini-2.0-flash = fastest & free. gemini-1.5-pro = best quality.",
    )

    model_info = {
        "gemini-2.0-flash": ("⚡ Fastest · Free tier · Recommended", "#39d0d8"),
        "gemini-1.5-flash": ("🚀 Fast · Great quality · Free tier", "#f0a500"),
        "gemini-1.5-pro": ("🧠 Best quality · 1M context · Limited free", "#58a6ff"),
    }
    info_text, info_color = model_info[model_choice]
    st.markdown(
        f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:0.68rem;'
        f'color:{info_color};margin-top:4px;">{info_text}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)
    load_btn = st.button("⬤ Load Video", type="primary", use_container_width=True)

    if st.session_state.video_loaded:
        st.markdown("<div style='margin-top:0.5rem'></div>", unsafe_allow_html=True)
        if st.button("↺ New Video", type="secondary", use_container_width=True):
            for k, v in defaults.items():
                st.session_state[k] = v
            st.rerun()

    # ── How it works ──
    st.markdown('<div class="sidebar-label" style="margin-top:1.5rem">ℹ️ How It Works</div>', unsafe_allow_html=True)
    steps = [
        "Fetch transcript + timestamps",
        "Chunk text with overlap",
        "Embed via Google models/embedding-001",
        "Store in FAISS vector index",
        "Your question → top-k retrieval",
        "Gemini answers from context only",
    ]
    for i, step in enumerate(steps, 1):
        st.markdown(
            f'<div class="how-step"><div class="step-num">{i}</div>{step}</div>',
            unsafe_allow_html=True,
        )

# ── Load Video Logic ──────────────────────────────────────────────────────────
if load_btn:
    if not st.session_state.gemini_key:
        st.error("⚠️ Please enter your Gemini API key in the sidebar.")
    elif not youtube_url:
        st.error("⚠️ Please paste a YouTube URL.")
    else:
        video_id = extract_video_id(youtube_url)
        if not video_id:
            st.error("❌ Could not parse video ID. Please check the URL.")
        else:
            with st.spinner("Fetching transcript & building vector index…"):
                try:
                    chunks = fetch_transcript(video_id)
                    if not chunks:
                        st.error("No transcript found. The video may not have captions enabled.")
                    else:
                        vs = build_vector_store(chunks, st.session_state.gemini_key)
                        st.session_state.vector_store = vs
                        st.session_state.video_id = video_id
                        st.session_state.transcript_chunks = chunks
                        st.session_state.video_loaded = True
                        st.session_state.messages = []
                        st.session_state.model_choice = model_choice
                        st.session_state.chunk_count = len(chunks)
                        st.session_state.video_duration = estimate_duration(chunks)
                        st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {e}")

# ── Main Content ──────────────────────────────────────────────────────────────

# ─ Video Loaded State ─
if st.session_state.video_loaded and st.session_state.video_id:
    vid = st.session_state.video_id
    thumb_url = f"https://img.youtube.com/vi/{vid}/mqdefault.jpg"
    watch_url = f"https://www.youtube.com/watch?v={vid}"

    st.markdown(f"""
    <div class="video-card">
        <img class="video-thumb" src="{thumb_url}" alt="thumbnail" />
        <div class="video-info">
            <div class="video-url">
                <a href="{watch_url}" target="_blank" style="color:#58a6ff;text-decoration:none;">
                    youtube.com/watch?v={vid}
                </a>
            </div>
            <div class="video-stats">
                <span class="stat-pill">📦 {st.session_state.chunk_count} chunks</span>
                <span class="stat-pill">⏱ ~{st.session_state.video_duration}</span>
                <span class="gem-badge">✦ {st.session_state.model_choice}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Render chat history ──
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                st.markdown('<div class="ts-section-label">▸ Jump to in video</div>', unsafe_allow_html=True)
                badges_html = ""
                for src in msg["sources"][:6]:
                    ts = format_timestamp(src)
                    url = f"https://www.youtube.com/watch?v={st.session_state.video_id}&t={src}s"
                    badges_html += f'<a class="ts-badge" href="{url}" target="_blank">▶ {ts}</a>'
                st.markdown(f'<div style="display:flex;flex-wrap:wrap;gap:4px;margin-top:2px">{badges_html}</div>', unsafe_allow_html=True)

    # ── Chat input ──
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
                        model=st.session_state.model_choice,
                        chat_history=st.session_state.messages[:-1],
                    )
                    st.markdown(answer)

                    if sources:
                        st.markdown('<div class="ts-section-label">▸ Jump to in video</div>', unsafe_allow_html=True)
                        badges_html = ""
                        for src in sources[:6]:
                            ts = format_timestamp(src)
                            url = f"https://www.youtube.com/watch?v={st.session_state.video_id}&t={src}s"
                            badges_html += f'<a class="ts-badge" href="{url}" target="_blank">▶ {ts}</a>'
                        st.markdown(f'<div style="display:flex;flex-wrap:wrap;gap:4px;margin-top:2px">{badges_html}</div>', unsafe_allow_html=True)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    })
                except Exception as e:
                    err = f"⚠️ Error: {e}"
                    st.error(err)
                    st.session_state.messages.append({"role": "assistant", "content": err, "sources": []})

# ─ Empty / Welcome State ─
else:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">🎬</div>
        <div class="empty-title">Ready to chat with any YouTube video</div>
        <div class="empty-desc">
            Enter your Gemini API key and a YouTube URL<br>
            in the sidebar, then click <strong style="color:#e6edf3">Load Video</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    cards = [
        ("🔑", "Free API Key", "Get Gemini API key from Google AI Studio — no credit card needed"),
        ("⚡", "Any Video", "Works with any YouTube video that has captions or auto-subtitles"),
        ("🌐", "Cloud Ready", "Deploy to Streamlit Cloud — store key in Secrets, zero config"),
    ]
    for col, (icon, title, desc) in zip([col1, col2, col3], cards):
        with col:
            st.markdown(f"""
            <div style="
                background:var(--bg-elevated);
                border:1px solid var(--border);
                border-radius:var(--radius-md);
                padding:18px 16px;
                text-align:center;
            ">
                <div style="font-size:1.8rem;margin-bottom:10px">{icon}</div>
                <div style="font-family:'Manrope',sans-serif;font-weight:700;
                            font-size:0.9rem;color:var(--text-primary);margin-bottom:6px">{title}</div>
                <div style="font-family:'IBM Plex Mono',monospace;font-size:0.72rem;
                            color:var(--text-muted);line-height:1.6">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
