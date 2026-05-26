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
    initial_sidebar_state="auto",
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


# =========================================================
# GLOBAL CSS
# =========================================================
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500;600;700;800&display=swap');

/* =====================================================
   ROOT
===================================================== */
:root {
    --bg:        #050816;
    --surface:   #0f172a;
    --gold:      #f5a623;
    --gold-soft: rgba(245,166,35,0.12);
    --cyan:      #4ecdc4;
    --text:      #f8fafc;
    --muted:     #94a3b8;
    --success:   #6bcb77;
    --border:    rgba(255,255,255,0.08);
}

/* =====================================================
   APP BACKGROUND
===================================================== */
html, body, .stApp {
    background:
        radial-gradient(circle at top left, #172554 0%, transparent 28%),
        radial-gradient(circle at bottom right, #581c87 0%, transparent 28%),
        linear-gradient(135deg, #050816 0%, #0b1120 100%) !important;

    background-attachment: fixed !important;

    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
}

/* =====================================================
   HIDE DEFAULT STREAMLIT
===================================================== */
#MainMenu,
footer {
    visibility: hidden;
}

/* =====================================================
   SCROLLBAR
===================================================== */
::-webkit-scrollbar {
    width: 5px;
}

::-webkit-scrollbar-thumb {
    background: #1e293b;
    border-radius: 999px;
}

::-webkit-scrollbar-thumb:hover {
    background: #334155;
}

/* =====================================================
   FIXED SIDEBAR TOGGLE
===================================================== */

/* BOTH STATES */
button[kind="header"],
[data-testid="collapsedControl"] {

    position: fixed !important;

    top: 18px !important;
    left: 14px !important;

    z-index: 999999 !important;

    width: 42px !important;
    height: 42px !important;

    display: flex !important;
    align-items: center !important;
    justify-content: center !important;

    visibility: visible !important;
    opacity: 1 !important;

    border-radius: 12px !important;

    border: 1px solid rgba(245,166,35,0.35) !important;

    background: rgba(5,8,22,0.82) !important;

    backdrop-filter: blur(10px) !important;

    box-shadow: 0 0 16px rgba(245,166,35,0.20) !important;

    transition: all 0.2s ease !important;
}

/* HOVER */
button[kind="header"]:hover,
[data-testid="collapsedControl"]:hover {

    background: rgba(245,166,35,0.15) !important;

    border-color: var(--gold) !important;

    transform: scale(1.05) !important;

    box-shadow: 0 0 24px rgba(245,166,35,0.40) !important;
}

/* ICON */
button[kind="header"] svg,
[data-testid="collapsedControl"] svg {

    width: 18px !important;
    height: 18px !important;

    fill: var(--gold) !important;
}

/* REMOVE STREAMLIT HIDE */
[data-testid="collapsedControl"] {
    transform: translateX(0px) !important;
}

/* KEEP HEADER VISIBLE */
header {
    visibility: visible !important;
    background: transparent !important;
}

/* =====================================================
   SIDEBAR
===================================================== */
[data-testid="stSidebar"] {

    background: rgba(8,12,28,0.95) !important;

    border-right: 1px solid rgba(245,166,35,0.12) !important;

    backdrop-filter: blur(20px) !important;
}

[data-testid="stSidebar"] .block-container {
    padding: 1.8rem 1.1rem 2rem !important;
}

/* =====================================================
   SIDEBAR BRAND
===================================================== */
.sb-brand {
    font-family: 'Space Mono', monospace;
    font-size: 0.63rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: 0.25rem;
}

.sb-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.08rem;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 1.6rem;
    line-height: 1.3;
}

.sb-label {
    font-size: 0.65rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--muted);
    font-family: 'Space Mono', monospace;
    margin: 1.1rem 0 0.4rem;
    display: block;
}

/* =====================================================
   FILE UPLOADER
===================================================== */
[data-testid="stFileUploader"] {

    background: rgba(255,255,255,0.03) !important;

    border: 1.5px dashed rgba(245,166,35,0.30) !important;

    border-radius: 14px !important;

    padding: 4px 8px !important;

    transition: all 0.2s !important;
}

[data-testid="stFileUploader"]:hover {

    border-color: var(--gold) !important;

    background: rgba(245,166,35,0.04) !important;
}

/* =====================================================
   TEXTAREA
===================================================== */
.stTextArea textarea {

    background: rgba(255,255,255,0.04) !important;

    border: 1.5px solid rgba(255,255,255,0.10) !important;

    border-radius: 14px !important;

    color: var(--text) !important;

    caret-color: var(--gold) !important;

    transition: border-color 0.2s !important;
}

.stTextArea textarea:focus {

    border-color: var(--gold) !important;

    box-shadow: 0 0 0 3px rgba(245,166,35,0.14) !important;
}

/* =====================================================
   SELECTBOX
===================================================== */
.stSelectbox [data-baseweb="select"] > div:first-child {

    background: rgba(255,255,255,0.04) !important;

    border: 1.5px solid rgba(255,255,255,0.10) !important;

    border-radius: 14px !important;

    color: var(--text) !important;
}

/* =====================================================
   BUTTON
===================================================== */
.stButton > button {

    width: 100% !important;

    border: none !important;

    border-radius: 14px !important;

    padding: 0.80rem 1rem !important;

    font-family: 'Space Mono', monospace !important;

    font-size: 0.76rem !important;

    font-weight: 700 !important;

    letter-spacing: 0.10em !important;

    text-transform: uppercase !important;

    color: #050816 !important;

    background: linear-gradient(
        135deg,
        #f5a623 0%,
        #ffb703 100%
    ) !important;

    box-shadow:
        0 0 18px rgba(245,166,35,0.30),
        0 0 40px rgba(245,166,35,0.10) !important;

    transition: all 0.22s ease !important;
}

.stButton > button:hover {

    transform: translateY(-2px) !important;

    box-shadow:
        0 0 28px rgba(245,166,35,0.55),
        0 0 60px rgba(245,166,35,0.18) !important;
}

/* =====================================================
   MAIN AREA
===================================================== */
.main .block-container {

    padding: 1.8rem 3rem 110px !important;

    max-width: 1080px !important;
}

/* =====================================================
   TOPBAR
===================================================== */
.topbar {

    background: rgba(5,8,22,0.58);

    backdrop-filter: blur(16px);

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

.topbar-sub {
    color: var(--muted);
    font-size: 0.88rem;
}

/* =====================================================
   HERO
===================================================== */
.hero {
    padding: 0.5rem 0 2rem;
}

.hero-badge {

    display: inline-block;

    padding: 7px 14px;

    background: rgba(245,166,35,0.10);

    border: 1px solid rgba(245,166,35,0.28);

    border-radius: 999px;

    color: var(--gold);

    font-size: 0.68rem;

    letter-spacing: 0.16em;

    font-family: 'Space Mono', monospace;

    margin-bottom: 1.2rem;
}

.hero h1 {

    font-size: 3.1rem;

    font-weight: 800;

    line-height: 1.06;

    margin: 0 0 1rem;

    background: linear-gradient(
        90deg,
        #ffffff 30%,
        #f5a623 100%
    );

    -webkit-background-clip: text;

    -webkit-text-fill-color: transparent;
}

.hero p {

    color: var(--muted);

    line-height: 1.8;

    font-size: 0.96rem;

    max-width: 660px;
}

/* =====================================================
   METRIC CARD
===================================================== */
.metric-card {

    background: rgba(17,24,39,0.55);

    border: 1px solid rgba(255,255,255,0.07);

    border-radius: 20px;

    padding: 1.15rem 1.3rem;

    backdrop-filter: blur(12px);
}

.metric-label {

    color: var(--muted);

    font-size: 0.70rem;

    text-transform: uppercase;

    letter-spacing: 0.10em;

    margin-bottom: 0.45rem;

    font-family: 'Space Mono', monospace;
}

.metric-value {

    font-size: 1.55rem;

    font-weight: 700;

    color: var(--text);
}

.metric-value.sm {
    font-size: 0.90rem;
}

/* =====================================================
   STATUS
===================================================== */
.status-ready {

    display: inline-flex;

    align-items: center;

    gap: 9px;

    padding: 9px 18px;

    border-radius: 999px;

    background: rgba(107,203,119,0.12);

    border: 1px solid rgba(107,203,119,0.25);

    color: var(--success);

    font-size: 0.76rem;

    font-weight: 700;

    font-family: 'Space Mono', monospace;
}

.pulse-dot {

    width: 9px;
    height: 9px;

    border-radius: 50%;

    background: var(--success);

    animation: pulse 1.8s ease-in-out infinite;
}

@keyframes pulse {

    0%, 100% {
        transform: scale(1);
        opacity: 1;
    }

    50% {
        transform: scale(1.4);
        opacity: 0.5;
    }
}

/* =====================================================
   CHAT MESSAGE
===================================================== */
[data-testid="stChatMessage"] {

    border-radius: 22px !important;

    padding: 1.1rem 1.3rem !important;

    border: 1px solid rgba(255,255,255,0.06) !important;

    backdrop-filter: blur(14px) !important;

    margin-bottom: 0.9rem !important;
}

/* =====================================================
   CHAT INPUT
===================================================== */
[data-testid="stChatInput"] {

    background: rgba(15,23,42,0.70) !important;

    border: 1.5px solid rgba(255,255,255,0.09) !important;

    border-radius: 20px !important;

    backdrop-filter: blur(14px) !important;
}

/* =====================================================
   EMPTY STATE
===================================================== */
.empty-state {

    text-align: center;

    padding: 5rem 2rem 3rem;
}

.empty-icon {

    font-size: 5rem;

    display: block;

    margin-bottom: 1rem;
}

/* =====================================================
   FOOTER
===================================================== */
.footer {

    position: fixed;

    bottom: 0;
    left: 0;

    width: 100%;

    text-align: center;

    padding: 12px 0 8px;

    font-size: 0.78rem;

    color: #475569;

    background: linear-gradient(
        to top,
        rgba(5,8,22,0.98) 55%,
        transparent
    );

    backdrop-filter: blur(6px);

    z-index: 200;
}

.footer a {

    color: var(--gold);

    text-decoration: none;

    margin: 0 9px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:

    st.markdown("""
    <div class="sb-brand">Document Intelligence</div>
    <div class="sb-title">
        Knowledge<br>Ingestion
    </div>
    """, unsafe_allow_html=True)

    # FILE UPLOAD
    st.markdown(
        '<span class="sb-label">📎 Upload Files</span>',
        unsafe_allow_html=True
    )

    files = st.file_uploader(
        "Upload PDF or TXT",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    # TEXT INPUT
    st.markdown(
        '<span class="sb-label">✍️ Paste Text</span>',
        unsafe_allow_html=True
    )

    manual_text = st.text_area(
        "Paste text",
        placeholder="Paste raw text here...",
        height=130,
        label_visibility="collapsed",
    )

    # MODEL
    st.markdown(
        '<span class="sb-label">🤖 Select Model</span>',
        unsafe_allow_html=True
    )

    llm_choice = st.selectbox(
        "Choose model",
        options=[
            "llama-3.1-8b-instant",
            "qwen3-32b",
            "gpt-oss-120b",
            "gemini-2.5-flash-lite",
            "gemini-2.5-flash",
        ],
        label_visibility="collapsed",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # PROCESS
    if st.button("⚡ Process & Embed"):

        if not files and not manual_text.strip():
            st.warning("⚠️ Upload files or paste text first.")

        else:

            with st.spinner("Building vector database..."):

                if os.path.exists("chroma_db"):
                    shutil.rmtree("chroma_db", ignore_errors=True)

                docs = load_files(files, manual_text)

                create_vector_db(docs)

                st.session_state.vectordb_ready = True
                st.session_state.chat_history = []
                st.session_state.active_model = llm_choice
                st.session_state.doc_count = len(docs)

            st.success("✅ Vector database ready!")

    st.markdown("<br><br>", unsafe_allow_html=True)

    st.markdown("""
    <div style="
        text-align:center;
        color:#2d3f58;
        font-size:0.63rem;
        font-family:'Space Mono',monospace;
        letter-spacing:0.10em;
        line-height:1.9;
    ">
        LANGCHAIN · CHROMA<br>
        HUGGINGFACE · STREAMLIT
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# TOPBAR
# =========================================================
st.markdown("""
<div class="topbar">
    <div class="topbar-logo">🧠 RAG AI</div>
    <div class="topbar-sub">
        Multi-File Document Intelligence
    </div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# HERO
# =========================================================
st.markdown("""
<div class="hero">
    <div class="hero-badge">
        ✦ AI DOCUMENT INTELLIGENCE
    </div>

    <h1>
        Multi-File RAG<br>Chatbot
    </h1>

    <p>
        Upload PDFs, TXT files, or paste raw text —
        then chat with your documents using semantic
        retrieval and state-of-the-art LLMs.
    </p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# METRICS
# =========================================================
c1, c2, c3 = st.columns(3)

with c1:

    count = len(files) if files else 0

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">
            📄 Documents
        </div>

        <div class="metric-value">
            {count}
        </div>
    </div>
    """, unsafe_allow_html=True)

with c2:

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">
            🧩 Chunks
        </div>

        <div class="metric-value">
            {st.session_state.doc_count}
        </div>
    </div>
    """, unsafe_allow_html=True)

with c3:

    model_display = (
        st.session_state.active_model
        if st.session_state.active_model
        else llm_choice
    )

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">
            🤖 Active Model
        </div>

        <div class="metric-value sm">
            {model_display}
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# =========================================================
# CHAT
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

    vectordb = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )

    retriever = vectordb.as_retriever(
        search_kwargs={"k": 4}
    )

    llm = get_llm(llm_choice)

    # HISTORY
    for user_msg, bot_msg in st.session_state.chat_history:

        st.chat_message(
            "user",
            avatar="👤"
        ).write(user_msg)

        st.chat_message(
            "assistant",
            avatar="🧠"
        ).write(bot_msg)

    # INPUT
    query = st.chat_input(
        "Ask a question about your documents..."
    )

    if query:

        st.chat_message(
            "user",
            avatar="👤"
        ).write(query)

        docs_retrieved = retriever.invoke(query)

        context = "\n\n".join(
            d.page_content for d in docs_retrieved
        )

        prompt = f"""
You are a helpful AI assistant.

Answer ONLY from the provided context.

If the answer is not in the context,
clearly state that the information is unavailable.

Context:
{context}

Question:
{query}
"""

        with st.spinner("Thinking..."):

            response = llm.invoke(prompt)

            answer = response.content

        st.chat_message(
            "assistant",
            avatar="🧠"
        ).write(answer)

        st.session_state.chat_history.append(
            (query, answer)
        )

else:

    st.markdown("""
    <div class="empty-state">

        <span class="empty-icon">
            📂
        </span>

        <div class="empty-title">
            No Documents Indexed Yet
        </div>

        <div class="empty-sub">
            Upload PDF / TXT files or paste raw text
            in the sidebar,<br>

            then click
            <strong>⚡ Process &amp; Embed</strong>
            to begin.
        </div>

    </div>
    """, unsafe_allow_html=True)


# =========================================================
# FOOTER
# =========================================================
st.markdown("""
<div class="footer">

    © 2026 <strong>Shreeyansh Asati</strong> ·

    <a href="https://www.linkedin.com/in/shreeyansh-asati-18shreey/"
       target="_blank">
       LinkedIn
    </a>

    ·

    <a href="https://github.com/SHREEYANSHGIT"
       target="_blank">
       GitHub
    </a>

</div>
""", unsafe_allow_html=True)
