import os
import streamlit as st
from dotenv import load_dotenv  # ✅ Added import
from pypdf import PdfReader
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai

# Config
DB_DIR = "./chroma_db"
COLLECTION_NAME = "papers"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
N_RESULTS = 5

# Gemini setup
load_dotenv()  
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash")  # ✅ Changed to stable model

# Functions
def extract_text_from_pdf(path):
    reader = PdfReader(path)
    return "\n".join([page.extract_text() or "" for page in reader.pages])

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

@st.cache_resource
def get_embedding_fn_and_client():
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL_NAME
    )
    client = chromadb.PersistentClient(path=DB_DIR)
    return embed_fn, client

# Streamlit UI
st.set_page_config(page_title="Local RAG with Gemini", page_icon="🧠")
st.title("🧠 Local RAG — Gemini API for QA")

embedder, client = get_embedding_fn_and_client()
try:
    collection = client.get_collection(COLLECTION_NAME, embedding_function=embedder)  # ✅ pass embedder
except:
    collection = client.create_collection(COLLECTION_NAME, embedding_function=embedder)

# Upload & Index
st.header("1) Upload and Index PDF")
uploaded = st.file_uploader("Upload PDF", type=["pdf"])
if uploaded:
    save_path = os.path.join("uploads", uploaded.name)
    os.makedirs("uploads", exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(uploaded.getbuffer())

    if st.button("📚 Index PDF"):
        raw_text = extract_text_from_pdf(save_path)
        chunks = chunk_text(raw_text)
        ids = [f"{uploaded.name}-{i}" for i in range(len(chunks))]
        metas = [{"source": uploaded.name, "chunk": i} for i in range(len(chunks))]
        collection.add(documents=chunks, metadatas=metas, ids=ids)
        st.success(f"Indexed {len(chunks)} chunks.")

# Ask questions
st.header("2) Ask Questions")
question = st.text_input("Your question:")
if st.button("🔎 Retrieve & Ask Gemini") and question.strip():
    res = collection.query(query_texts=[question], n_results=N_RESULTS)
    docs = res.get("documents", [[]])[0]
    if not docs:
        st.warning("No results found.")
    else:
        context = "\n".join(docs)
        prompt = f"Answer the question based only on the context below.\nContext:\n{context}\n\nQuestion: {question}"
        response = model.generate_content(prompt)
        st.subheader("Answer")
        st.write(response.text)
        st.subheader("Retrieved Passages")
        for i, doc in enumerate(docs, 1):
            st.write(f"**Passage {i}:** {doc}")
