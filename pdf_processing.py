from pypdf import PdfReader
import streamlit as st
from config import CHUNK_SIZE, CHUNK_OVERLAP

def extract_text_from_pdf(path):
    try:
        reader = PdfReader(path)
        return "\n".join([page.extract_text() or "" for page in reader.pages])
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = text.replace("\r", " ").replace("\n", " ")
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i:i+chunk_size]
        if chunk.strip():
            chunks.append(chunk)
        i += chunk_size - overlap
    return chunks