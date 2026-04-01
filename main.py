import streamlit as st
from utils import (
    extract_video_id,
    fetch_transcript,
    summarize_with_gemini,
    fetch_video_metadata,
    generate_key_points,
    generate_quiz,
    export_summary_as_txt,
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="YT Summarizer",
    page_icon="▶",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject custom CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

  :root {
    --bg-primary:   #0a0a0f;
    --bg-card:      #111118;
    --bg-elevated:  #16161f;
    --accent:       #7c3aed;
    --accent-glow:  #7c3aed55;
    --accent-light: #a78bfa;
    --success:      #10b981;
    --warning:      #f59e0b;
    --danger:       #ef4444;
    --text-primary: #f1f5f9;
    --text-muted:   #64748b;
    --border:       #1e1e2e;
    --border-hover: #7c3aed66;
  }

  html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-primary) !important;
    font-family: 'DM Sans', sans-serif;
    color: var(--text-primary);
  }

  [data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border);
  }
  [data-testid="stSidebar"] * { color: var(--text-primary) !important; }

  #MainMenu, footer, header { visibility: hidden; }

  h1, h2, h3 { font-family: 'Space Mono', monospace !important; }

  .stTextInput > div > div > input,
  .stTextArea > div > textarea,
  .stSelectbox > div > div {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    transition: border-color 0.2s;
  }
  .stTextInput > div > div > input:focus,
  .stTextArea > div > textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-glow) !important;
  }

  .stButton > button {
    background: linear-gradient(135deg, var(--accent), #5b21b6) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.85rem !important;
    padding: 0.55rem 1.4rem !important;
    letter-spacing: 0.04em;
    transition: opacity 0.2s, box-shadow 0.2s !important;
  }
  .stButton > button:hover {
    opacity: 0.88 !important;
    box-shadow: 0 0 18px var(--accent-glow) !important;
  }

  .stDownloadButton > button {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--accent) !important;
    color: var(--accent-light) !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
  }

  .stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card) !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 4px;
  }
  .stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-muted) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.78rem !important;
    border-radius: 6px 6px 0 0 !important;
    padding: 0.5rem 1.1rem !important;
  }
  .stTabs [aria-selected="true"] {
    background: var(--bg-elevated) !important;
    color: var(--accent-light) !important;
    border-bottom: 2px solid var(--accent) !important;
  }

  .streamlit-expanderHeader {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.82rem !important;
  }
  .streamlit-expanderContent {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
    border-radius: 0 0 8px 8px !important;
  }

  [data-testid="metric-container"] {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 1rem !important;
  }
  [data-testid="metric-container"] label {
    color: var(--text-muted) !important;
    font-size: 0.75rem !important;
  }
  [data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--accent-light) !important;
    font-family: 'Space Mono', monospace !important;
  }

  .stAlert { border-radius: 8px !important; border-left-width: 3px !important; }
  .stSpinner > div { border-top-color: var(--accent) !important; }

  .yt-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s;
  }
  .yt-card:hover { border-color: var(--border-hover); }

  .hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--accent-light), #c4b5fd);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
    margin-bottom: 0.3rem;
  }
  .hero-sub {
    color: var(--text-muted);
    font-size: 0.95rem;
    margin-bottom: 2rem;
  }

  .badge {
    display: inline-block;
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    color: var(--accent-light);
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    padding: 2px 10px;
    border-radius: 20px;
    margin-right: 6px;
  }

  .divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 1.4rem 0;
  }

  pre, code {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    color: var(--accent-light) !important;
    font-family: 'Space Mono', monospace !important;
  }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ▶ YT Summarizer")
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── API key ───────────────────────────────────────────────────────────────
    api_key: str | None = None
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        st.success("API key loaded ✓", icon="🔑")
    except (KeyError, FileNotFoundError):
        st.error("Add GEMINI_API_KEY to .streamlit/secrets.toml", icon="⚠")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("**Settings**")

    summary_style = st.selectbox(
        "Summary style",
        ["Concise", "Detailed", "Bullet Points", "Executive Brief"],
        index=0,
    )
    summary_language = st.selectbox(
        "Output language",
        ["English", "Hindi", "Spanish", "French", "German", "Japanese"],
        index=0,
    )
    max_tokens = st.slider("Max output tokens", 256, 4096, 1024, step=128)
    include_timestamps = st.toggle("Include timestamps", value=False)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("**🍪 YouTube Cookies**")
    st.caption(
        "If transcript fetching is blocked, upload your `cookies.txt` "
        "(Netscape format). Export it with the "
        "[Get cookies.txt](https://chromewebstore.google.com/detail/get-cookiestxt-locally/cclelndahbckbenkjhflpdbgdldlbecc) "
        "Chrome extension while logged into YouTube."
    )

    # ── Cookies: file uploader takes priority, then secrets fallback ──────────
    cookies_txt: str | None = None

    uploaded_cookies = st.file_uploader(
        "cookies.txt (optional)",
        type=["txt"],
        label_visibility="collapsed",
    )
    if uploaded_cookies is not None:
        try:
            cookies_txt = uploaded_cookies.read().decode("utf-8", errors="ignore")
            if cookies_txt.strip():
                st.success("Cookies loaded ✓", icon="🍪")
            else:
                st.warning("Uploaded file appears empty.", icon="⚠")
                cookies_txt = None
        except Exception as exc:
            st.error(f"Could not read cookies file: {exc}", icon="⚠")
            cookies_txt = None
    else:
        # Fallback: read from Streamlit secrets
        try:
            secret_cookies = st.secrets["YT_COOKIES"]
            if secret_cookies and secret_cookies.strip():
                cookies_txt = secret_cookies
                st.success("Cookies loaded from secrets ✓", icon="🍪")
        except (KeyError, FileNotFoundError):
            cookies_txt = None

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown(
        '<span class="badge">Gemini 2.0 Flash</span>'
        '<span class="badge">v1.0.0</span>',
        unsafe_allow_html=True,
    )


# ── Main layout ───────────────────────────────────────────────────────────────
st.markdown('<p class="hero-title">YouTube Summarizer</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">Paste any YouTube URL → get instant AI-powered insights.</p>',
    unsafe_allow_html=True,
)

col_input, col_btn = st.columns([5, 1], vertical_alignment="bottom")
with col_input:
    url = st.text_input(
        "YouTube URL",
        placeholder="https://www.youtube.com/watch?v=...",
        label_visibility="collapsed",
    )
with col_btn:
    run = st.button("⚡ Analyse", use_container_width=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── Processing ────────────────────────────────────────────────────────────────
if run:
    if not api_key:
        st.error("No API key found. Add GEMINI_API_KEY to your Streamlit secrets.")
        st.stop()
    if not url.strip():
        st.warning("Please enter a YouTube URL.")
        st.stop()

    video_id = extract_video_id(url)
    if not video_id:
        st.error("Could not parse a valid YouTube video ID from that URL.")
        st.stop()

    # ── Fetch metadata ────────────────────────────────────────────────────────
    with st.spinner("Fetching video metadata…"):
        meta = fetch_video_metadata(video_id)

    if meta:
        m1, m2, m3, m4 = st.columns(4)
        raw_title = meta.get("title", "—")
        m1.metric("▶ Title", raw_title[:30] + "…" if len(raw_title) > 30 else raw_title)
        m2.metric("👤 Channel",  meta.get("author", "—"))
        m3.metric("⏱ Duration", meta.get("duration", "—"))
        m4.metric("👁 Views",    meta.get("views", "—"))

        thumb = meta.get("thumbnail")
        if thumb:
            with st.expander("🖼 Thumbnail preview", expanded=False):
                st.image(thumb, use_container_width=False, width=480)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Fetch transcript ──────────────────────────────────────────────────────
    with st.spinner("Extracting transcript…"):
        transcript, transcript_error = fetch_transcript(
            video_id,
            include_timestamps=include_timestamps,
            cookies_txt=cookies_txt,
        )

    if transcript_error:
        st.error(f"Transcript error: {transcript_error}")
        st.stop()

    st.session_state["transcript"] = transcript
    st.session_state["video_id"]   = video_id
    st.session_state["meta"]       = meta or {}

    # ── Summarise ─────────────────────────────────────────────────────────────
    with st.spinner("Generating summary with Gemini…"):
        summary = summarize_with_gemini(
            transcript=transcript,
            api_key=api_key,
            style=summary_style,
            language=summary_language,
            max_tokens=max_tokens,
        )

    with st.spinner("Extracting key points…"):
        key_points = generate_key_points(transcript, api_key, max_tokens)

    with st.spinner("Building quiz questions…"):
        quiz = generate_quiz(transcript, api_key)

    st.session_state["summary"]    = summary
    st.session_state["key_points"] = key_points
    st.session_state["quiz"]       = quiz


# ── Results ───────────────────────────────────────────────────────────────────
if "summary" in st.session_state:
    summary    = st.session_state["summary"]
    key_points = st.session_state["key_points"]
    quiz       = st.session_state["quiz"]
    transcript = st.session_state["transcript"]

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📝 Summary", "🎯 Key Points", "❓ Quiz", "📄 Transcript"]
    )

    with tab1:
        st.markdown('<div class="yt-card">', unsafe_allow_html=True)
        st.markdown(summary)
        st.markdown('</div>', unsafe_allow_html=True)

        txt_export = export_summary_as_txt(
            title=st.session_state.get("meta", {}).get("title", "Video"),
            summary=summary,
            key_points=key_points,
        )
        st.download_button(
            "⬇ Download Summary (.txt)",
            data=txt_export,
            file_name="summary.txt",
            mime="text/plain",
        )

    with tab2:
        if isinstance(key_points, list):
            for i, point in enumerate(key_points, 1):
                st.markdown(
                    f'<div class="yt-card"><span class="badge">{i}</span> {point}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown('<div class="yt-card">', unsafe_allow_html=True)
            st.markdown(key_points)
            st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        if isinstance(quiz, list):
            for i, item in enumerate(quiz, 1):
                with st.expander(f"Q{i}: {item.get('question', 'Question')}", expanded=False):
                    st.markdown(f"**Answer:** {item.get('answer', '—')}")
                    if item.get("explanation"):
                        st.caption(item["explanation"])
        else:
            st.markdown(quiz)

    with tab4:
        with st.expander("Full transcript", expanded=False):
            st.text_area(
                "",
                value=transcript,
                height=420,
                label_visibility="collapsed",
            )
        st.download_button(
            "⬇ Download Transcript (.txt)",
            data=transcript,
            file_name="transcript.txt",
            mime="text/plain",
        )
