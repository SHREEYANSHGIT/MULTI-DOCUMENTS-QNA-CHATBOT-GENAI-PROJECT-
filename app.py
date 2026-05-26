import streamlit as st
import os
import shutil
from ingest import load_files, create_vector_db
from llm_router import get_llm
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# ---------- PAGE CONFIGURATION ----------
st.set_page_config(
    page_title="Multi-File RAG Chatbot", 
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- CUSTOM CSS (Modern UI Elements) ----------
st.markdown("""
    <style>
    /* Gradient Title */
    .main-title {
        background: linear-gradient(45deg, #4A90E2, #50E3C2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: -10px;
        padding-top: 20px;
    }
    .sub-title {
        text-align: center;
        color: #A0AEC0;
        font-size: 1.1rem;
        font-weight: 400;
        margin-bottom: 40px;
    }
    
    /* Modern Button Styling */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        border: 1px solid #4A90E2;
        background-color: transparent;
        color: #4A90E2;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #4A90E2;
        color: white;
        box-shadow: 0 4px 12px rgba(74, 144, 226, 0.3);
        transform: translateY(-2px);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #1E1E2F;
    }
    
    /* Adjust main padding to prevent footer overlap */
    .block-container {
        padding-bottom: 100px;
    }
    
    /* Sleek Footer */
    .modern-footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background: rgba(15, 15, 25, 0.8);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        color: #A0AEC0;
        text-align: center;
        font-size: 14px;
        padding: 12px;
        z-index: 100;
    }
    .modern-footer a {
        color: #50E3C2;
        text-decoration: none;
        margin: 0 10px;
        font-weight: 600;
        transition: color 0.2s;
    }
    .modern-footer a:hover {
        color: #4A90E2;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- SESSION STATE ----------
if "vectordb_ready" not in st.session_state:
    st.session_state.vectordb_ready = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------- MAIN HEADER ----------
st.markdown('<h1 class="main-title">Multi-File RAG Chatbot</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Intelligent document analysis powered by advanced LLMs</p>', unsafe_allow_html=True)

# ---------- SIDEBAR ----------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712139.png", width=60) # Placeholder AI Icon
    st.header("⚙️ Configuration")
    st.markdown("Upload your knowledge base to begin.")
    
    with st.container():
        files = st.file_uploader(
            "📄 Upload TXT / PDF",
            type=["txt", "pdf"],
            accept_multiple_files=True,
            help="Select one or multiple files to analyze."
        )
        
        manual_text = st.text_area(
            "✍️ Or paste raw text", 
            placeholder="Paste any additional context here...",
            height=120
        )
        
    st.divider()
    
    llm_choice = st.selectbox(
        "🧠 Select Intelligence Model",
        [
            "llama-3.1-8b-instant",
            "qwen3-32b",
            "gpt-oss-120b",
            "gemini-2.5-flash-lite",
            "gemini-2.5-flash"
        ],
        help="Choose the underlying LLM to power your responses."
    )
    
    st.divider()
    
    if st.button("🚀 Process Data & Initialize"):
        if not files and not manual_text.strip():
            st.error("⚠️ Please upload files or paste text to proceed.")
        else:
            with st.status("Initializing Knowledge Base...", expanded=True) as status:
                st.write("🗑️ Cleaning previous database...")
                # 🔥 FULLY DELETE OLD EMBEDDINGS
                if os.path.exists("chroma_db"):
                    shutil.rmtree("chroma_db", ignore_errors=True)
                
                st.write("📚 Loading and chunking documents...")
                docs = load_files(files, manual_text)
                
                st.write("🧩 Generating vector embeddings...")
                create_vector_db(docs)
                
                st.session_state.chat_history = []
                st.session_state.vectordb_ready = True
                status.update(label="✅ Database Ready!", state="complete", expanded=False)
            st.toast("Initialization complete! You can now ask questions.", icon="🎉")

# ---------- CHAT INTERFACE ----------
if st.session_state.vectordb_ready:
    # Initialize connection to DB and LLM
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectordb = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    llm = get_llm(llm_choice)

    # 1. Render previous chat history so it doesn't disappear on app rerun
    if not st.session_state.chat_history:
        st.info("👋 **Welcome!** Your documents are loaded. Ask a question below to get started.")
    
    for q, a in st.session_state.chat_history:
        with st.chat_message("user", avatar="👤"):
            st.write(q)
        with st.chat_message("assistant", avatar="🤖"):
            st.write(a)

    # 2. Handle new user input
    query = st.chat_input("Message the RAG Chatbot...")
    if query:
        # Display user query immediately
        with st.chat_message("user", avatar="👤"):
            st.write(query)
            
        # Fetch answer with a loading spinner
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Searching documents & generating response..."):
                docs = retriever.invoke(query)
                context = "\n\n".join(doc.page_content for doc in docs)
                
                prompt = f"""
You are a helpful assistant.
Answer ONLY from the context below.
Context:
{context}
Question:
{query}
"""
                response = llm.invoke(prompt)
                answer = response.content
                st.write(answer)
                
        # Save to history
        st.session_state.chat_history.append((query, answer))

else:
    # Empty State Display
    st.write("")
    st.write("")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info("👈 **Awaiting Data Input:** Upload documents in the sidebar and click **Process Data** to start chatting.")

# ---------- MODERN FOOTER ----------
st.markdown(
    """
    <div class="modern-footer">
        © 2026 <b>Developed by Shreeyansh Asati</b> |
        <a href="https://www.linkedin.com/in/shreeyansh-asati-18shreey/" target="_blank">🔗 LinkedIn</a> |
        <a href="https://github.com/SHREEYANSHGIT" target="_blank">💻 GitHub</a>
    </div>
    """,
    unsafe_allow_html=True
)
