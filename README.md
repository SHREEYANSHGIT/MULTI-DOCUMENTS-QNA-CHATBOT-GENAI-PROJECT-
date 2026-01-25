
📂 MULTI-FILE RAG CHATBOT (GEN AI PROJECT)
============================================================

👨‍💻 Developed by : Shreeyansh Asati  
🔗 LinkedIn     : https://www.linkedin.com/in/shreeyansh-asati-18shreey/  
💻 GitHub       : https://github.com/SHREEYANSHGIT  
🌐 APP          : https://multi-documents-qna-chatbot-shreeyansh.streamlit.app/


🚀 PROJECT OVERVIEW
============================================================

This project is a **Multi-File Question Answering Chatbot** built using
**Retrieval-Augmented Generation (RAG)**.

The chatbot allows users to:<br>
✅ Upload multiple files (PDF, TXT)<br>
✅ Paste manual text input<br>
✅ Ask questions in natural language<br>
✅ Get answers ONLY from the uploaded data<br>
✅ Replace old knowledge when new files are uploaded<br>

The system uses **Transformer embeddings**, **ChromaDB**, and **LLMs
(Gemini & Groq)** for high-quality and fast responses.


🧠 TECH STACK
============================================================

🔹 Frontend - Streamlit<br>
🔹 LLMs (Free / Fast)<br>

      • Gemini :  Gemini 2.5 Flash-lite ,Gemini 2.5 Flash
      • Groq : llama-3.1-8b-instant , qwen/qwen3-32b ,openai/gpt-oss-120b 
🔹 Frameworks & Libraries : LangChain , Sentence Transformers , Pandas<br>
🔹 Embeddings : all-MiniLM-L6-v2 (Transformer-based)<br>
🔹 Vector Database : ChromaDB (Persistent Vector Store)<br>

📁 SUPPORTED INPUT TYPES
============================================================

📄 PDF Files (.pdf)<br>
📄 Text Files (.txt)<br>
✍️ Manual Text Input (Paste inside app)

All inputs are combined into a **single knowledge base** per upload.

⚙️ SYSTEM ARCHITECTURE
============================================================

<img width="2562" height="1673" alt="⚙️ SYSTEM ARCHITECTURE - visual selection (1)" src="https://github.com/user-attachments/assets/c03ae9e0-f9c3-4772-9177-3393cf353177" />


🔄 DATA REPLACEMENT LOGIC (IMPORTANT)
============================================================

When the user clicks **"Process Data"**:

✔ Old ChromaDB embeddings are DELETED<br>
✔ New embeddings are created from uploaded files<br>
✔ Chat memory is reset<br>
✔ The chatbot answers ONLY from new data<br>

This prevents:
❌ Old file contamination
❌ Incorrect retrieval
❌ Mixed answers


🗂️ PROJECT STRUCTURE
============================================================

<img width="663" height="304" alt="image" src="https://github.com/user-attachments/assets/cef80843-8ca7-46bd-92c6-d407dba09379" />


🔐 ENVIRONMENT SETUP
============================================================

Python Version:
✔ Python 3.10 / 3.11 (Recommended)

Install Dependencies:
--------------------------------
    pip install -r requirements.txt
--------------------------------

Add API Keys:
Create file `.streamlit/secrets.toml`

--------------------------------
    GOOGLE_API_KEY = "your_gemini_key"
    GROQ_API_KEY   = "your_groq_key"
--------------------------------


▶️ RUN LOCALLY
============================================================
    streamlit run app.py


☁️ STREAMLIT CLOUD DEPLOYMENT
============================================================

1️⃣ Push code to GitHub (exclude secrets & chroma_db)<br>
2️⃣ Go to https://streamlit.io/cloud<br>
3️⃣ Click "New App"<br>
4️⃣ Select repository<br>
5️⃣ Set main file → app.py<br>
6️⃣ Add secrets in Streamlit Cloud<br>
7️⃣ Deploy 🚀<br>

🧠 KEY FEATURES
============================================================

✅ Multi-file RAG system<br>
✅ Old embeddings replaced on new upload<br>
✅ Fast inference with Groq<br>
✅ Strong reasoning with Gemini<br>
✅ Clean Streamlit UI<br>
✅ Production-safe architecture<br>
✅ Interview & hackathon ready<br>


📜 LICENSE
============================================================

© 2026 Shreeyansh Asati  
All rights reserved.
