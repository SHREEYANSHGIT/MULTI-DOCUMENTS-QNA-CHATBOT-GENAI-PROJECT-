import streamlit as st
import os
import shutil

from ingest import load_files, create_vector_db
from llm_router import get_llm

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="RAG AI · Shreeyansh",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# SESSION STATE
# =========================================================
if "vectordb_ready" not in st.session_state:
    st.session_state.vectordb_ready = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "active_model" not in st.session_state:
    st.session_state.active_model = ""

if "doc_count" not in st.session_state:
    st.session_state.doc_count = 0

if "sidebar_visible" not in st.session_state:
    st.session_state.sidebar_visible = True


# =========================================================
# GLOBAL CSS
# =========================================================
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── ROOT ── */
:root {
    --bg:         #050816;
    --surface:    #0f172a;
    --surface2:   #111827;
    --border:     rgba(255,255,255,0.08);
    --gold:       #f5a623;
    --gold-soft:  rgba(245,166,35,0.12);
    --cyan:       #4ecdc4;
    --text:       #f8fafc;
    --muted:      #94a3b8;
    --success:    #6bcb77;
}

/* ── APP BACKGROUND ── */
html, body, .stApp {
    background:
        radial-gradient(circle at top left,    #172554 0%, transparent 28%),
        radial-gradient(circle at bottom right, #581c87 0%, transparent 28%),
        linear-gradient(135deg, #050816 0%, #0b1120 100%) !important;
    background-attachment: fixed !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── HIDE STREAMLIT CHROME ── */
#MainMenu, header, footer { visibility: hidden; }

/* ── SCROLLBAR ── */
::-webkit-scrollbar              { width: 5px; }
::-webkit-scrollbar-thumb        { background: #1e293b; border-radius: 999px; }
::-webkit-scrollbar-thumb:hover  { background: #334155; }

/* ══════════════════════════════════════════
   SIDEBAR TOGGLE BUTTON
══════════════════════════════════════════ */
.sidebar-toggle {
    position: fixed;
    left: 20px;
    top: 20px;
    z-index: 999;
    background: linear-gradient(135deg, #f5a623 0%, #ffb703 100%);
    border: none;
    border-radius: 12px;
    padding: 10px 16px;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    color: #050816;
    cursor: pointer;
    transition: all 0.22s ease;
    box-shadow: 0 0 18px rgba(245,166,35,0.35);
}
.sidebar-toggle:hover {
    transform: translateY(-2px);
    box-shadow: 0 0 28px rgba(245,166,35,0.55);
}

/* ══════════════════════════════════════════
   SIDEBAR
══════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background: rgba(10,15,30,0.92) !important;
    border-right: 1px solid rgba(245,166,35,0.12) !important;
    transition: transform 0.3s ease !important;
}

[data-testid="stSidebar"] > div:first-child {
    padding: 0 !important;
}

[data-testid="stSidebar"] .block-container {
    padding: 1.8rem 1.1rem 2rem !important;
}

/* Hide sidebar */
[data-testid="stSidebar"][aria-expanded="false"] {
    transform: translateX(-100%) !important;
}

/* Sidebar brand */
.sb-brand {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: 0.25rem;
}
.sb-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 1.8rem;
    line-height: 1.3;
}

/* Section label */
.sb-label {
    font-size: 0.67rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--muted);
    font-family: 'Space Mono', monospace;
    margin: 1.1rem 0 0.45rem;
    display: block;
}

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1.5px dashed rgba(245,166,35,0.30) !important;
    border-radius: 14px !important;
    padding: 6px 10px !important;
    transition: border-color 0.2s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--gold) !important;
    background: rgba(245,166,35,0.05) !important;
}
[data-testid="stFileUploader"] label {
    color: var(--muted) !important;
    font-size: 0.82rem !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] {
    color: var(--muted) !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
}

/* ── TEXTAREA ── */
.stTextArea textarea {
    background: rgba(255,255,255,0.04) !important;
    border: 1.5px solid rgba(255,255,255,0.10) !important;
    border-radius: 14px !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.88rem !important;
    caret-color: var(--gold) !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
    padding: 10px 14px !important;
}
.stTextArea textarea:focus {
    border-color: var(--gold) !important;
    box-shadow: 0 0 0 3px rgba(245,166,35,0.14) !important;
    outline: none !important;
}
.stTextArea textarea::placeholder {
    color: #475569 !important;
}
.stTextArea label {
    color: var(--muted) !important;
    font-size: 0.82rem !important;
}

/* ── SELECTBOX ── */
.stSelectbox > label {
    color: var(--muted) !important;
    font-size: 0.82rem !important;
}
.stSelectbox [data-baseweb="select"] > div:first-child {
    background: rgba(255,255,255,0.04) !important;
    border: 1.5px solid rgba(255,255,255,0.10) !important;
    border-radius: 14px !important;
    color: var(--text) !important;
    font-size: 0.88rem !important;
    transition: border-color 0.2s !important;
}
.stSelectbox [data-baseweb="select"] > div:first-child:hover {
    border-color: var(--gold) !important;
}
/* Dropdown menu */
[data-baseweb="popover"] [data-baseweb="menu"] {
    background: #0f172a !important;
    border: 1px solid rgba(245,166,35,0.20) !important;
    border-radius: 14px !important;
}
[data-baseweb="popover"] li {
    color: var(--text) !important;
    font-size: 0.88rem !important;
}
[data-baseweb="popover"] li:hover {
    background: rgba(245,166,35,0.10) !important;
}

/* ── BUTTON ── */
.stButton > button {
    width: 100% !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 0.78rem 1rem !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.10em !important;
    text-transform: uppercase !important;
    color: #050816 !important;
    background: linear-gradient(135deg, #f5a623 0%, #ffb703 100%) !important;
    box-shadow: 0 0 18px rgba(245,166,35,0.35), 0 0 40px rgba(245,166,35,0.10) !important;
    transition: all 0.22s ease !important;
    margin-top: 0.4rem !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 0 28px rgba(245,166,35,0.55), 0 0 60px rgba(245,166,35,0.18) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* ══════════════════════════════════════════
   MAIN AREA
══════════════════════════════════════════ */
.main .block-container {
    padding: 1.8rem 3rem 110px !important;
    max-width: 1080px !important;
}

/* ── TOPBAR ── */
.topbar {
    background: rgba(5,8,22,0.60);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 18px;
    padding: 13px 22px;
    margin-bottom: 2.2rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.topbar-logo {
    font-family: 'Space Mono', monospace;
    color: var(--gold);
    font-size: 0.95rem;
    letter-spacing: 0.18em;
    font-weight: 700;
}
.topbar-sub { color: var(--muted); font-size: 0.88rem; }

/* ── HERO ── */
.hero { padding: 0.5rem 0 2rem; }
.hero-badge {
    display: inline-block;
    padding: 7px 14px;
    background: rgba(245,166,35,0.10);
    border: 1px solid rgba(245,166,35,0.28);
    border-radius: 999px;
    color: var(--gold);
    font-size: 0.70rem;
    letter-spacing: 0.16em;
    font-family: 'Space Mono', monospace;
    margin-bottom: 1.2rem;
}
.hero h1 {
    font-size: 3.2rem;
    font-weight: 800;
    line-height: 1.05;
    margin: 0 0 1rem;
    background: linear-gradient(90deg, #ffffff 30%, #f5a623 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero p {
    color: var(--muted);
    line-height: 1.8;
    font-size: 0.97rem;
    max-width: 680px;
    margin: 0;
}

/* ── METRIC CARDS ── */
.metric-card {
    background: rgba(17,24,39,0.55);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 20px;
    padding: 1.2rem 1.3rem;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: rgba(245,166,35,0.25); }
.metric-label {
    color: var(--muted);
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.10em;
    margin-bottom: 0.5rem;
    font-family: 'Space Mono', monospace;
}
.metric-value {
    font-size: 1.55rem;
    font-weight: 700;
    color: var(--text);
}
.metric-value.sm { font-size: 0.92rem; }

/* ── STATUS BADGE ── */
.status-ready {
    display: inline-flex;
    align-items: center;
    gap: 9px;
    padding: 9px 18px;
    border-radius: 999px;
    background: rgba(107,203,119,0.12);
    border: 1px solid rgba(107,203,119,0.25);
    color: var(--success);
    font-size: 0.78rem;
    font-weight: 700;
    font-family: 'Space Mono', monospace;
    letter-spacing: 0.06em;
    margin-bottom: 1.4rem;
}
.pulse-dot {
    width: 9px;
    height: 9px;
    border-radius: 50%;
    background: var(--success);
    box-shadow: 0 0 10px var(--success);
    animation: pulse 1.8s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { transform: scale(1);   opacity: 1;   }
    50%       { transform: scale(1.4); opacity: 0.5; }
}

/* ── CHAT MESSAGES ── */
[data-testid="stChatMessage"] {
    border-radius: 22px !important;
    padding: 1.1rem 1.3rem !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    backdrop-filter: blur(14px) !important;
    margin-bottom: 0.9rem !important;
    transition: transform 0.18s ease, border-color 0.18s ease !important;
}
[data-testid="stChatMessage"]:hover {
    transform: translateY(-2px) !important;
    border-color: rgba(245,166,35,0.18) !important;
}
/* user bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: linear-gradient(135deg,
        rgba(245,166,35,0.13) 0%,
        rgba(245,166,35,0.04) 100%) !important;
}
/* assistant bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background: rgba(255,255,255,0.025) !important;
}

/* ── CHAT INPUT ── */
[data-testid="stChatInput"] {
    background: rgba(15,23,42,0.70) !important;
    border: 1.5px solid rgba(255,255,255,0.09) !important;
    border-radius: 20px !important;
    backdrop-filter: blur(14px) !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--gold) !important;
    box-shadow: 0 0 0 3px rgba(245,166,35,0.12) !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: var(--text) !important;
}

/* ── ALERTS ── */
[data-testid="stAlert"] {
    border-radius: 14px !important;
    font-size: 0.88rem !important;
}

/* ── SPINNER ── */
.stSpinner > div { border-top-color: var(--gold) !important; }

/* ── EMPTY STATE ── */
.empty-state {
    text-align: center;
    padding: 5rem 2rem 3rem;
}
.empty-icon {
    font-size: 5rem;
    display: block;
    margin-bottom: 1rem;
    animation: float 4s ease-in-out infinite;
}
@keyframes float {
    0%, 100% { transform: translateY(0);    }
    50%       { transform: translateY(-14px); }
}
.empty-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 0.55rem;
}
.empty-sub {
    color: var(--muted);
    line-height: 1.8;
    font-size: 0.9rem;
}

/* ── FOOTER ── */
.footer {
    position: fixed;
    bottom: 0; left: 0;
    width: 100%;
    text-align: center;
    padding: 12px 0 8px;
    font-size: 0.80rem;
    color: #475569;
    background: linear-gradient(to top, rgba(5,8,22,0.98) 55%, transparent);
    backdrop-filter: blur(6px);
    z-index: 200;
    font-family: 'Space Mono', monospace;
    letter-spacing: 0.04em;
}
.footer a {
    color: var(--gold);
    text-decoration: none;
    margin: 0 9px;
    transition: color 0.2s;
}
.footer a:hover { color: var(--cyan); }

/* ── DIVIDER ── */
hr {
    border: none !important;
    border-top: 1px solid rgba(255,255,255,0.06) !important;
    margin: 1.6rem 0 !important;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# SIDEBAR TOGGLE BUTTON
# =========================================================
def toggle_sidebar():
    st.session_state.sidebar_visible = not st.session_state.sidebar_visible

if st.button("📂 Show Upload Panel", key="toggle_sidebar", help="Show/hide file upload panel"):
    toggle_sidebar()
    st.rerun()

# =========================================================
# SIDEBAR
# =========================================================
if st.session_state.sidebar_visible:
    with st.sidebar:

        # Brand
        st.markdown("""
        <div class="sb-brand">Document Intelligence</div>
        <div class="sb-title">Knowledge<br>Ingestion</div>
        """, unsafe_allow_html=True)

        # ── File upload ──
        st.markdown('<span class="sb-label">📎 Upload Files</span>', unsafe_allow_html=True)
        files = st.file_uploader(
            "Upload PDF or TXT",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            label_visibility="visible",
            key="file_uploader"
        )

        # ── Paste text ──
        st.markdown('<span class="sb-label">✍️ Paste Text</span>', unsafe_allow_html=True)
        manual_text = st.text_area(
            "Or paste raw text here",
            placeholder="Paste any raw text content...",
            height=130,
            label_visibility="visible",
            key="manual_text"
        )

        # ── Model selector ──
        st.markdown('<span class="sb-label">🤖 Select Model</span>', unsafe_allow_html=True)
        llm_choice = st.selectbox(
            "Choose your LLM",
            options=[
                "llama-3.1-8b-instant",
                "qwen3-32b",
                "gpt-oss-120b",
                "gemini-2.5-flash-lite",
                "gemini-2.5-flash",
            ],
            label_visibility="visible",
            key="model_selector"
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Process button ──
        if st.button("⚡ Process & Embed", key="process_button"):
            if not files and not manual_text.strip():
                st.warning("⚠️ Upload files or paste text first.")
            else:
                with st.spinner("Building vector database..."):
                    if os.path.exists("chroma_db"):
                        shutil.rmtree("chroma_db", ignore_errors=True)

                    docs = load_files(files, manual_text)
                    create_vector_db(docs)

                    st.session_state.vectordb_ready = True
                    st.session_state.chat_history   = []
                    st.session_state.active_model   = llm_choice
                    st.session_state.doc_count      = len(docs)
                    
                    # Auto-hide sidebar after processing
                    st.session_state.sidebar_visible = False

                st.success("✅ Vector database ready!")
                st.rerun()

        # Sidebar footer
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="
            text-align:center;
            color:#2d3f58;
            font-size:0.65rem;
            font-family:'Space Mono',monospace;
            letter-spacing:0.10em;
            line-height:1.8;
        ">
            LANGCHAIN · CHROMA<br>HUGGINGFACE · STREAMLIT
        </div>
        """, unsafe_allow_html=True)
else:
    # Set default values when sidebar is hidden
    files = []
    manual_text = ""
    llm_choice = st.session_state.get("active_model", "llama-3.1-8b-instant")


# =========================================================
# TOPBAR
# =========================================================
st.markdown("""
<div class="topbar">
    <div class="topbar-logo">🧠 RAG AI</div>
    <div class="topbar-sub">Multi-File Document Intelligence</div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# HERO
# =========================================================
st.markdown("""
<div class="hero">
    <div class="hero-badge">✦ AI DOCUMENT INTELLIGENCE</div>
    <h1>Multi-File RAG<br>Chatbot</h1>
    <p>
        Upload PDFs, TXT files, or paste raw text — then chat with your
        documents using semantic retrieval and state-of-the-art LLMs.
    </p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# METRIC CARDS
# =========================================================
c1, c2, c3 = st.columns(3)

with c1:
    # Get file count from session state or current files
    if st.session_state.sidebar_visible and 'files' in locals():
        file_count = len(files) if files else 0
    else:
        file_count = 0
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">📄 Documents</div>
        <div class="metric-value">{file_count}</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">🧩 Chunks</div>
        <div class="metric-value">{st.session_state.doc_count}</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    model_display = st.session_state.active_model if st.session_state.active_model else "llama-3.1-8b-instant"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">🤖 Active Model</div>
        <div class="metric-value sm">{model_display}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# =========================================================
# CHAT SECTION
# =========================================================
if st.session_state.vectordb_ready:

    st.markdown("""
    <div class="status-ready">
        <div class="pulse-dot"></div>
        INDEX READY — ASK ANYTHING
    </div>
    """, unsafe_allow_html=True)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectordb  = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    llm       = get_llm(llm_choice)

    # Replay history
    for user_msg, bot_msg in st.session_state.chat_history:
        st.chat_message("user",      avatar="👤").write(user_msg)
        st.chat_message("assistant", avatar="🧠").write(bot_msg)

    # Input
    query = st.chat_input("Ask a question about your documents...")

    if query:
        st.chat_message("user", avatar="👤").write(query)

        docs_retrieved = retriever.invoke(query)
        context = "\n\n".join(d.page_content for d in docs_retrieved)

        prompt = f"""You are a helpful AI assistant.
Answer ONLY from the provided context.
If the answer is not in the context, clearly state that the information is unavailable.

Context:
{context}

Question:
{query}
"""
        with st.spinner("Thinking..."):
            response = llm.invoke(prompt)
            answer   = response.content

        st.chat_message("assistant", avatar="🧠").write(answer)
        st.session_state.chat_history.append((query, answer))

else:

    st.markdown("""
    <div class="empty-state">
        <span class="empty-icon">📂</span>
        <div class="empty-title">No Documents Indexed Yet</div>
        <div class="empty-sub">
            Click the <strong>📂 Show Upload Panel</strong> button above,<br>
            upload PDF / TXT files or paste raw text,<br>
            then click <strong>⚡ Process &amp; Embed</strong> to begin.
        </div>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# FOOTER
# =========================================================
st.markdown("""
<div class="footer">
    © 2026 &nbsp;<strong>Shreeyansh Asati</strong>&nbsp;·
    <a href="https://www.linkedin.com/in/shreeyansh-asati-18shreey/" target="_blank">LinkedIn</a>·
    <a href="https://github.com/SHREEYANSHGIT" target="_blank">GitHub</a>
</div>
""", unsafe_allow_html=True)
