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


# =========================================================
# GLOBAL CSS
# =========================================================
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500;600;700;800&display=swap');

/* =========================================================
   ROOT
========================================================= */
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

/* =========================================================
   APP BACKGROUND
========================================================= */
html, body, .stApp {
    background:
        radial-gradient(circle at top left, #172554 0%, transparent 28%),
        radial-gradient(circle at bottom right, #581c87 0%, transparent 28%),
        linear-gradient(135deg, #050816 0%, #0b1120 100%) !important;

    background-attachment: fixed !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
}

/* =========================================================
   HIDE DEFAULT STREAMLIT
========================================================= */
#MainMenu,
header,
footer {
    visibility: hidden;
}

/* =========================================================
   SCROLLBAR
========================================================= */
::-webkit-scrollbar {
    width: 5px;
}

::-webkit-scrollbar-thumb {
    background: #1e293b;
    border-radius: 999px;
}

/* =========================================================
   FIX SIDEBAR TOGGLE BUTTON
========================================================= */

/* collapsed button */
[data-testid="collapsedControl"] {
    position: fixed !important;
    top: 16px !important;
    left: 16px !important;

    width: 44px !important;
    height: 44px !important;

    display: flex !important;
    align-items: center !important;
    justify-content: center !important;

    visibility: visible !important;
    opacity: 1 !important;

    z-index: 999999 !important;

    background: rgba(10,14,30,0.92) !important;
    border: 1px solid rgba(245,166,35,0.35) !important;
    border-radius: 14px !important;

    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;

    box-shadow:
        0 0 18px rgba(245,166,35,0.22),
        0 0 40px rgba(245,166,35,0.10) !important;

    transition: all 0.22s ease !important;
}

/* hover */
[data-testid="collapsedControl"]:hover {
    transform: scale(1.06) !important;

    border-color: #f5a623 !important;

    background: rgba(245,166,35,0.12) !important;

    box-shadow:
        0 0 26px rgba(245,166,35,0.45),
        0 0 50px rgba(245,166,35,0.18) !important;
}

/* icon */
[data-testid="collapsedControl"] svg {
    width: 20px !important;
    height: 20px !important;
    fill: #f5a623 !important;
}

/* prevent hidden */
[data-testid="collapsedControl"][aria-expanded="false"] {
    display: flex !important;
    opacity: 1 !important;
    visibility: visible !important;
}

/* open state button */
button[kind="header"] {
    opacity: 1 !important;
    visibility: visible !important;
}

/* =========================================================
   SIDEBAR
========================================================= */
[data-testid="stSidebar"] {
    background: rgba(8,12,28,0.95) !important;
    border-right: 1px solid rgba(245,166,35,0.12) !important;
    backdrop-filter: blur(20px) !important;
}

[data-testid="stSidebar"] .block-container {
    padding: 1.8rem 1.1rem 2rem !important;
}

/* =========================================================
   SIDEBAR TEXT
========================================================= */
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

/* =========================================================
   FILE UPLOADER
========================================================= */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1.5px dashed rgba(245,166,35,0.30) !important;
    border-radius: 14px !important;
    padding: 4px 8px !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: var(--gold) !important;
}

/* =========================================================
   TEXTAREA
========================================================= */
.stTextArea textarea {
    background: rgba(255,255,255,0.04) !important;
    border: 1.5px solid rgba(255,255,255,0.10) !important;
    border-radius: 14px !important;
    color: var(--text) !important;
}

/* =========================================================
   SELECTBOX
========================================================= */
.stSelectbox [data-baseweb="select"] > div:first-child {
    background: rgba(255,255,255,0.04) !important;
    border: 1.5px solid rgba(255,255,255,0.10) !important;
    border-radius: 14px !important;
    color: var(--text) !important;
}

/* =========================================================
   BUTTON
========================================================= */
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
}

/* =========================================================
   MAIN AREA
========================================================= */
.main .block-container {
    padding: 1.8rem 3rem 110px !important;
    max-width: 1080px !important;
}

/* =========================================================
   HERO
========================================================= */
.hero h1 {
    font-size: 3.1rem;
    font-weight: 800;

    background: linear-gradient(
        90deg,
        #ffffff 30%,
        #f5a623 100%
    );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* =========================================================
   CHAT INPUT
========================================================= */
[data-testid="stChatInput"] {
    background: rgba(15,23,42,0.70) !important;

    border: 1.5px solid rgba(255,255,255,0.09) !important;

    border-radius: 20px !important;
}

/* =========================================================
   FOOTER
========================================================= */
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

    z-index: 200;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# JS FIX FOR SIDEBAR BUTTON
# =========================================================
st.markdown("""
<script>

const fixSidebar = () => {

    const btn = window.parent.document.querySelector(
        '[data-testid="collapsedControl"]'
    );

    if (btn) {
        btn.style.display = "flex";
        btn.style.visibility = "visible";
        btn.style.opacity = "1";
        btn.style.zIndex = "999999";
    }
};

setInterval(fixSidebar, 500);

</script>
""", unsafe_allow_html=True)


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:

    st.markdown("""
    <div class="sb-brand">Document Intelligence</div>
    <div class="sb-title">Knowledge<br>Ingestion</div>
    """, unsafe_allow_html=True)

    # FILES
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

    # TEXT
    st.markdown(
        '<span class="sb-label">✍️ Paste Text</span>',
        unsafe_allow_html=True
    )

    manual_text = st.text_area(
        "Paste text",
        placeholder="Paste raw text content...",
        height=130,
        label_visibility="collapsed",
    )

    # MODEL
    st.markdown(
        '<span class="sb-label">🤖 Select Model</span>',
        unsafe_allow_html=True
    )

    llm_choice = st.selectbox(
        "Choose LLM",
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


# =========================================================
# HERO
# =========================================================
st.markdown("""
<div class="hero">
    <h1>Multi-File RAG Chatbot</h1>

    <p>
        Upload PDFs, TXT files, or raw text and chat with your
        documents using semantic retrieval + LLMs.
    </p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# METRICS
# =========================================================
c1, c2, c3 = st.columns(3)

with c1:
    st.metric("📄 Documents", len(files) if files else 0)

with c2:
    st.metric("🧩 Chunks", st.session_state.doc_count)

with c3:
    active_model = (
        st.session_state.active_model
        if st.session_state.active_model
        else llm_choice
    )

    st.metric("🤖 Active Model", active_model)


# =========================================================
# CHAT SECTION
# =========================================================
if st.session_state.vectordb_ready:

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    llm = get_llm(llm_choice)

    # HISTORY
    for user_msg, bot_msg in st.session_state.chat_history:

        st.chat_message("user", avatar="👤").write(user_msg)

        st.chat_message("assistant", avatar="🧠").write(bot_msg)

    # INPUT
    query = st.chat_input(
        "Ask a question about your documents..."
    )

    if query:

        st.chat_message("user", avatar="👤").write(query)

        docs_retrieved = retriever.invoke(query)

        context = "\n\n".join(
            d.page_content for d in docs_retrieved
        )

        prompt = f"""
You are a helpful AI assistant.

Answer ONLY from the provided context.

If answer is unavailable,
say clearly that information is not found.

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

    st.info(
        "Upload PDF/TXT files or paste text in sidebar "
        "then click Process & Embed."
    )


# =========================================================
# FOOTER
# =========================================================
st.markdown("""
<div class="footer">
    © 2026 <strong>Shreeyansh Asati</strong>
</div>
""", unsafe_allow_html=True)
