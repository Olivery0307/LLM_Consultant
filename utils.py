import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

@st.cache_resource
def get_llm():
    """Creates and caches the LLM client to prevent re-initialization."""
    print("Creating LLM client...")
    return ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0, convert_system_message_to_human=True)

@st.cache_resource
def get_embeddings_model():
    """Creates and caches the embeddings model client."""
    print("Creating embeddings model client...")
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")