import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from config import DB_DIR, COLLECTION_NAME, EMBED_MODEL_NAME

@st.cache_resource
def get_embedding_fn_and_client():
    try:
        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL_NAME
        )
        client = chromadb.PersistentClient(path=DB_DIR)
        return embed_fn, client
    except Exception as e:
        st.error(f"Error initializing embedding function: {str(e)}")
        return None, None

def clear_collection_and_create_new(client, embedder):
    """Clear existing collection and create a new one"""
    try:
        try:
            client.delete_collection(COLLECTION_NAME)
        except:
            pass
        
        collection = client.create_collection(COLLECTION_NAME, embedding_function=embedder)
        return collection
    except Exception as e:
        st.error(f"Error creating new collection: {str(e)}")
        return None

def get_current_paper_docs(collection, current_paper_name):
    """Get documents only from the current paper"""
    try:
        all_docs = collection.get(
            where={"source": current_paper_name}
        )
        return all_docs.get("documents", [])
    except Exception as e:
        st.error(f"Error retrieving current paper documents: {str(e)}")
        return []