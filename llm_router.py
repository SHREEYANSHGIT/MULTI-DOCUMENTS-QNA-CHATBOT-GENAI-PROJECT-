from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq


def get_llm(model_name: str):

    # -------- GEMINI --------
    if model_name == "gemini-2.5-flash-lite":
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            temperature=0.3
        )

    if model_name == "gemini-2.5-flash":
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3
        )

    # -------- GROQ --------
    if model_name == "llama-3.1-8b-instant":
        return ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)

    if model_name == "qwen3-32b":
        return ChatGroq(model="qwen/qwen3-32b", temperature=0.2)

    if model_name == "gpt-oss-120b":
        return ChatGroq(model="openai/gpt-oss-120b", temperature=0.15)

    raise ValueError("Invalid model selected")

