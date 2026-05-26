import streamlit as st
import os
import shutil
from ingest import load_files, create_vector_db
from llm_router import get_llm
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Multi-File RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Container */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.3);
        text-align: center;
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        font-weight: 400;
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
        border-right: 1px solid rgba(0,0,0,0.05);
    }
    
    /* Cards */
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border: 1px solid rgba(102, 126, 234, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .info-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.2);
    }
    
    .info-card h3 {
        color: #667eea;
        font-size: 1.2rem;
        margin-bottom: 0.5rem;
    }
    
    .info-card p {
        color: #6c757d;
        font-size: 0.9rem;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 0.75rem 2rem !important;
        border-radius: 50px !important;
        border: none !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        width: 100% !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.4) !important;
    }
    
    .stButton > button:active {
        transform: translateY(0) !important;
    }
    
    /* Chat Messages */
    .stChatMessage {
        background: white !important;
        border-radius: 20px !important;
        padding: 1rem 1.5rem !important;
        margin-bottom: 1rem !important;
        box-shadow: 0 5px 20px rgba(0,0,0,0.05) !important;
        border: 1px solid rgba(0,0,0,0.05) !important;
    }
    
    /* User Message */
    [data-testid="stChatMessage"][aria-label="user"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
    }
    
    /* Assistant Message */
    [data-testid="stChatMessage"][aria-label="assistant"] {
        background: #f8f9fa !important;
        border-left: 4px solid #667eea !important;
    }
    
    /* Chat Input */
    .stChatInput {
        background: white !important;
        border: 2px solid rgba(102, 126, 234, 0.2) !important;
        border-radius: 50px !important;
        padding: 1rem !important;
        box-shadow: 0 5px 20px rgba(0,0,0,0.05) !important;
    }
    
    .stChatInput:focus {
        border-color: #667eea !important;
        box-shadow: 0 5px 25px rgba(102, 126, 234, 0.2) !important;
    }
    
    /* File Uploader */
    .stFileUploader {
        background: white !important;
        border: 2px dashed rgba(102, 126, 234, 0.3) !important;
        border-radius: 15px !important;
        padding: 1.5rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stFileUploader:hover {
        border-color: #667eea !important;
        background: rgba(102, 126, 234, 0.02) !important;
    }
    
    /* Select Box */
    .stSelectbox > div > div {
        background: white !important;
        border: 2px solid rgba(102, 126, 234, 0.2) !important;
        border-radius: 50px !important;
    }
    
    /* Text Area */
    .stTextArea > div > div {
        background: white !important;
        border: 2px solid rgba(102, 126, 234, 0.2) !important;
        border-radius: 15px !important;
    }
    
    /* Success/Warning Messages */
    .stSuccess {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%) !important;
        border: none !important;
        border-radius: 15px !important;
        color: #1a1a1a !important;
        font-weight: 500 !important;
    }
    
    .stWarning {
        background: linear-gradient(135deg, #f6d365 0%, #fda085 100%) !important;
        border: none !important;
        border-radius: 15px !important;
        color: #1a1a1a !important;
        font-weight: 500 !important;
    }
    
    .stInfo {
        background: linear-gradient(135deg, #a1c4fd 0%, #c2e9fb 100%) !important;
        border: none !important;
        border-radius: 15px !important;
        color: #1a1a1a !important;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-color: #667eea transparent #764ba2 transparent !important;
    }
    
    /* Footer */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        backdrop-filter: blur(20px);
        color: #495057;
        text-align: center;
        font-size: 14px;
        padding: 15px;
        z-index: 100;
        border-top: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    .footer a {
        color: #667eea;
        text-decoration: none;
        margin: 0 10px;
        font-weight: 600;
        transition: color 0.3s ease;
    }
    
    .footer a:hover {
        color: #764ba2;
        text-decoration: none;
    }
    
    /* Emoji animations */
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }
    
    .animated-emoji {
        animation: float 3s ease-in-out infinite;
        display: inline-block;
    }
    
    /* Badge */
    .model-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.3rem 1rem;
        border-radius: 50px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }
    
    /* Divider */
    .custom-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent 0%, rgba(102, 126, 234, 0.3) 50%, transparent 100%);
        margin: 1.5rem 0;
    }
    
    /* Chat history counter */
    .chat-counter {
        background: rgba(102, 126, 234, 0.1);
        padding: 0.5rem 1rem;
        border-radius: 50px;
        font-size: 0.9rem;
        font-weight: 500;
        color: #667eea;
        text-align: center;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------- SESSION STATE ----------
if "vectordb_ready" not in st.session_state:
    st.session_state.vectordb_ready = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------- HEADER ----------
st.markdown("""
<div class="main-header">
    <h1>
        <span class="animated-emoji">📂</span> 
        Multi-File RAG Chatbot 
        <span class="animated-emoji">🤖</span>
    </h1>
    <p>Intelligent Document Analysis & Conversation</p>
</div>
""", unsafe_allow_html=True)

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    
    # Upload Section
    st.markdown("### 📤 Upload Documents")
    with st.container():
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        files = st.file_uploader(
            "Choose TXT or PDF files",
            type=["txt", "pdf"],
            accept_multiple_files=True,
            help="Upload your documents for analysis"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Text Input Section
    st.markdown("### ✍️ Or Paste Text")
    with st.container():
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        manual_text = st.text_area(
            "Enter your text here",
            height=120,
            placeholder="Paste your text content here..."
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Model Selection
    st.markdown("### 🤖 Select Model")
    with st.container():
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        llm_choice = st.selectbox(
            "Choose your AI model",
            [
                "llama-3.1-8b-instant",
                "qwen3-32b",
                "gpt-oss-120b",
                "gemini-2.5-flash-lite",
                "gemini-2.5-flash"
            ],
            help="Select the language model for answering questions"
        )
        st.markdown(f'<span class="model-badge">Active: {llm_choice}</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Process Button
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("🚀 Process & Embed", use_container_width=True):
            if not files and not manual_text.strip():
                st.warning("⚠️ Please upload files or paste text first.")
            else:
                with st.spinner("🔄 Creating intelligent embeddings..."):
                    # Delete old embeddings
                    if os.path.exists("chroma_db"):
                        shutil.rmtree("chroma_db", ignore_errors=True)
                    
                    docs = load_files(files, manual_text)
                    create_vector_db(docs)
                    st.session_state.chat_history = []
                    st.session_state.vectordb_ready = True
                    st.success("✅ Ready! Start asking questions.")
                    st.balloons()
    
    # Stats
    if st.session_state.chat_history:
        st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="chat-counter">
            💬 Questions asked: {len(st.session_state.chat_history)}
        </div>
        """, unsafe_allow_html=True)

# ---------- MAIN CHAT AREA ----------
if st.session_state.vectordb_ready:
    # Initialize embeddings and vector DB
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectordb = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    llm = get_llm(llm_choice)
    
    # Display chat history
    for i, (question, answer) in enumerate(st.session_state.chat_history):
        with st.chat_message("user", avatar="🧑"):
            st.markdown(f"**You:** {question}")
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(answer)
    
    # Chat input
    query = st.chat_input("💭 Ask anything about your documents...")
    
    if query:
        # Display user message
        with st.chat_message("user", avatar="🧑"):
            st.markdown(f"**You:** {query}")
        
        # Generate response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("🤔 Thinking..."):
                docs = retriever.invoke(query)
                context = "\n\n".join(doc.page_content for doc in docs)
                prompt = f"""
You are a helpful assistant.
Answer ONLY from the context below.
If the answer is not in the context, say "I cannot find this information in the provided documents."

Context:
{context}

Question:
{query}
"""
                response = llm.invoke(prompt)
                answer = response.content
                st.markdown(answer)
        
        # Save to history
        st.session_state.chat_history.append((query, answer))

else:
    # Welcome message when no documents are processed
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 3rem;'>
            <div style='font-size: 5rem; margin-bottom: 2rem;'>
                <span class="animated-emoji">📚</span>
                <span class="animated-emoji">🔍</span>
                <span class="animated-emoji">💡</span>
            </div>
            <h2 style='color: #667eea; margin-bottom: 1rem;'>Welcome to RAG Chatbot!</h2>
            <p style='color: #6c757d; font-size: 1.1rem; margin-bottom: 2rem;'>
                Upload your documents in the sidebar and start an intelligent conversation.
            </p>
            <div class='info-card'>
                <h3>🚀 Getting Started</h3>
                <p>1. Upload PDF or TXT files<br>
                   2. Or paste your text directly<br>
                   3. Click "Process & Embed"<br>
                   4. Start asking questions!</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ---------- FOOTER ----------
st.markdown("""
<div class="footer">
    <div style="display: flex; justify-content: center; align-items: center; gap: 20px;">
        <span>© 2026 <strong>Shreeyansh Asati</strong></span>
        <span style="color: #667eea;">|</span>
        <a href="https://www.linkedin.com/in/shreeyansh-asati-18shreey/" target="_blank">
            <span style="font-size: 1.1rem;">🔗</span> LinkedIn
        </a>
        <span style="color: #667eea;">|</span>
        <a href="https://github.com/SHREEYANSHGIT" target="_blank">
            <span style="font-size: 1.1rem;">💻</span> GitHub
        </a>
        <span style="color: #667eea;">|</span>
        <span style="color: #6c757d;">Powered by LangChain & Streamlit</span>
    </div>
</div>
""", unsafe_allow_html=True)
