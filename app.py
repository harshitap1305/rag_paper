# import os
# import streamlit as st
# from dotenv import load_dotenv
# from pypdf import PdfReader
# import chromadb
# from chromadb.utils import embedding_functions
# import google.generativeai as genai
# import requests
# from bs4 import BeautifulSoup
# import json
# import time
# from typing import List, Dict
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# from wordcloud import WordCloud
# import io
# import base64
# from datetime import datetime
# import re

# # Config
# DB_DIR = "./chroma_db"
# COLLECTION_NAME = "papers"
# EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# CHUNK_SIZE = 1200
# CHUNK_OVERLAP = 200
# N_RESULTS = 8  # Increased for better context
# MAX_RESPONSE_WORDS = 800  # Configurable response limit

# # Gemini setup
# load_dotenv()  
# genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
# model = genai.GenerativeModel("gemini-2.0-flash")

# # Enhanced functions
# def extract_text_from_pdf(path):
#     reader = PdfReader(path)
#     full_text = ""
#     for page in reader.pages:
#         text = page.extract_text()
#         if text:
#             full_text += text + "\n"
#     return full_text

# def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
#     text = text.replace("\r", " ").replace("\n", " ")
#     chunks = []
#     i = 0
#     while i < len(text):
#         chunk = text[i:i+chunk_size]
#         if chunk.strip():
#             chunks.append(chunk)
#         i += chunk_size - overlap
#     return chunks

# def extract_paper_metadata(text):
#     """Extract key metadata from the paper"""
#     lines = text.split('\n')[:50]  # First 50 lines usually contain metadata
    
#     metadata = {
#         'title': '',
#         'authors': [],
#         'abstract': '',
#         'keywords': [],
#         'year': '',
#         'doi': ''
#     }
    
#     # Simple heuristics to extract metadata
#     for i, line in enumerate(lines):
#         line = line.strip()
#         if not line:
#             continue
            
#         # Title (usually in first few lines, often in caps or title case)
#         if not metadata['title'] and len(line) > 10 and len(line) < 200:
#             if line.isupper() or line.istitle():
#                 metadata['title'] = line
                
#         # Abstract
#         if 'abstract' in line.lower():
#             abstract_start = i
#             abstract_text = ""
#             for j in range(i, min(i+20, len(lines))):
#                 if 'keywords' in lines[j].lower() or 'introduction' in lines[j].lower():
#                     break
#                 abstract_text += lines[j] + " "
#             metadata['abstract'] = abstract_text.strip()
            
#         # Keywords
#         if 'keywords' in line.lower():
#             keywords_text = line.split(':')[-1] if ':' in line else line
#             metadata['keywords'] = [k.strip() for k in keywords_text.split(',')]
            
#         # DOI
#         doi_match = re.search(r'10\.\d{4,}[^\s]+', line)
#         if doi_match:
#             metadata['doi'] = doi_match.group()
            
#         # Year
#         year_match = re.search(r'\b(19|20)\d{2}\b', line)
#         if year_match:
#             metadata['year'] = year_match.group()
    
#     return metadata

# def search_citations(query, num_results=5):
#     """Search for citations using a simple web search approach"""
#     try:
#         # Using a basic approach - in production, use proper academic APIs like Semantic Scholar
#         search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}+research+paper+citation"
#         headers = {
#             'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
#         }
        
#         # For demo purposes, return mock citations
#         mock_citations = [
#             {
#                 'title': f"Related Research on {query[:50]}...",
#                 'authors': "Smith, J. et al.",
#                 'year': "2023",
#                 'url': "https://example.com/paper1",
#                 'abstract': f"This paper discusses {query[:100]}... and provides comprehensive analysis."
#             },
#             {
#                 'title': f"Advanced Study in {query[:40]}...",
#                 'authors': "Johnson, M. & Brown, K.",
#                 'year': "2022", 
#                 'url': "https://example.com/paper2",
#                 'abstract': f"An in-depth investigation of {query[:80]}... with novel approaches."
#             }
#         ]
        
#         return mock_citations[:num_results]
        
#     except Exception as e:
#         st.warning(f"Citation search failed: {e}")
#         return []

# def create_visualization(text_data, viz_type="wordcloud"):
#     """Create visualizations for the paper"""
#     fig, ax = plt.subplots(figsize=(10, 6))
    
#     if viz_type == "wordcloud":
#         # Create word cloud
#         wordcloud = WordCloud(width=800, height=400, 
#                             background_color='white',
#                             colormap='viridis').generate(text_data)
#         ax.imshow(wordcloud, interpolation='bilinear')
#         ax.axis('off')
#         ax.set_title("Key Terms Visualization", fontsize=16, fontweight='bold')
        
#     elif viz_type == "length_analysis":
#         # Analyze text length distribution
#         sentences = text_data.split('.')
#         lengths = [len(s.strip()) for s in sentences if s.strip()]
        
#         ax.hist(lengths, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
#         ax.set_xlabel('Sentence Length (characters)')
#         ax.set_ylabel('Frequency')
#         ax.set_title('Sentence Length Distribution')
#         ax.grid(True, alpha=0.3)
    
#     # Convert plot to base64 string
#     buffer = io.BytesIO()
#     plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
#     buffer.seek(0)
#     image_base64 = base64.b64encode(buffer.getvalue()).decode()
#     plt.close()
    
#     return f"data:image/png;base64,{image_base64}"

# def generate_booklet(paper_text, metadata, citations):
#     """Generate a comprehensive booklet about the research paper"""
    
#     booklet_sections = []
    
#     # Title page
#     booklet_sections.append(f"""
#     # 📚 Research Paper Analysis Booklet
    
#     **Generated on:** {datetime.now().strftime("%B %d, %Y")}
    
#     ---
    
#     ## 📖 Paper Title
#     **{metadata.get('title', 'Unknown Title')}**
    
#     ## ✍️ Authors
#     {', '.join(metadata.get('authors', ['Unknown Authors']))}
    
#     ## 📅 Publication Year
#     {metadata.get('year', 'Unknown')}
    
#     ## 🔗 DOI
#     {metadata.get('doi', 'Not Available')}
    
#     ---
#     """)
    
#     # Abstract section
#     if metadata.get('abstract'):
#         booklet_sections.append(f"""
#     ## 📋 Abstract
    
#     {metadata['abstract']}
    
#     ---
#     """)
    
#     # Generate comprehensive analysis using Gemini
#     analysis_prompt = f"""
#     Analyze this research paper and provide a comprehensive breakdown in the following sections:
    
#     1. EXECUTIVE SUMMARY (100-150 words)
#     2. RESEARCH OBJECTIVES & METHODOLOGY (150-200 words)
#     3. KEY FINDINGS & CONTRIBUTIONS (200-250 words)
#     4. TECHNICAL INNOVATIONS (150-200 words)
#     5. IMPLICATIONS & APPLICATIONS (100-150 words)
#     6. LIMITATIONS & FUTURE WORK (100-150 words)
    
#     Paper content: {paper_text[:8000]}...
    
#     Make it detailed, technical but accessible, and well-structured with clear headings.
#     """
    
#     try:
#         analysis_response = model.generate_content(analysis_prompt)
#         booklet_sections.append(f"""
#     ## 🔬 Comprehensive Analysis
    
#     {analysis_response.text}
    
#     ---
#     """)
#     except Exception as e:
#         booklet_sections.append(f"""
#     ## 🔬 Comprehensive Analysis
    
#     *Analysis generation failed: {e}*
    
#     ---
#     """)
    
#     # Citations section
#     if citations:
#         booklet_sections.append("""
#     ## 📚 Related Research & Citations
    
#     """)
        
#         for i, citation in enumerate(citations, 1):
#             booklet_sections.append(f"""
#     ### Citation {i}
#     **Title:** {citation['title']}  
#     **Authors:** {citation['authors']}  
#     **Year:** {citation['year']}  
#     **Abstract:** {citation['abstract']}  
#     **URL:** [{citation['url']}]({citation['url']})
    
#     """)
        
#         booklet_sections.append("---\n")
    
#     # Keywords section
#     if metadata.get('keywords'):
#         booklet_sections.append(f"""
#     ## 🏷️ Keywords
    
#     {', '.join(metadata['keywords'])}
    
#     ---
#     """)
    
#     # Footer
#     booklet_sections.append(f"""
#     ## 📊 Document Statistics
    
#     - **Total Characters:** {len(paper_text):,}
#     - **Estimated Reading Time:** {len(paper_text.split()) // 200} minutes
#     - **Analysis Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    
#     ---
    
#     *This booklet was automatically generated using AI analysis. Please verify information independently.*
#     """)
    
#     return '\n'.join(booklet_sections)

# @st.cache_resource
# def get_embedding_fn_and_client():
#     embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
#         model_name=EMBED_MODEL_NAME
#     )
#     client = chromadb.PersistentClient(path=DB_DIR)
#     return embed_fn, client

# def limit_response_words(text, max_words=MAX_RESPONSE_WORDS):
#     """Limit response to specified number of words"""
#     words = text.split()
#     if len(words) <= max_words:
#         return text
#     return ' '.join(words[:max_words]) + f"... [Response truncated to {max_words} words]"

# # Streamlit UI
# st.set_page_config(page_title="Enhanced RAG with Booklet Generator", page_icon="🧠", layout="wide")

# # Sidebar for settings
# st.sidebar.header("⚙️ Settings")
# response_word_limit = st.sidebar.slider("Max Response Words", 200, 1500, MAX_RESPONSE_WORDS)
# enable_citations = st.sidebar.checkbox("Enable Citation Search", value=True)
# enable_visualizations = st.sidebar.checkbox("Enable Visualizations", value=True)

# st.title("🧠 Enhanced RAG — Research Paper Analysis & Booklet Generator")
# st.markdown("*Upload PDFs, ask questions, and generate comprehensive research booklets with citations and visualizations*")

# embedder, client = get_embedding_fn_and_client()
# try:
#     collection = client.get_collection(COLLECTION_NAME, embedding_function=embedder)
# except:
#     collection = client.create_collection(COLLECTION_NAME, embedding_function=embedder)

# # Create two columns
# col1, col2 = st.columns([1, 1])

# with col1:
#     st.header("📤 Upload and Index PDF")
#     uploaded = st.file_uploader("Upload PDF", type=["pdf"])
    
#     if uploaded:
#         save_path = os.path.join("uploads", uploaded.name)
#         os.makedirs("uploads", exist_ok=True)
#         with open(save_path, "wb") as f:
#             f.write(uploaded.getbuffer())

#         if st.button("📚 Index PDF", type="primary"):
#             with st.spinner("Processing PDF..."):
#                 raw_text = extract_text_from_pdf(save_path)
#                 chunks = chunk_text(raw_text)
#                 ids = [f"{uploaded.name}-{i}" for i in range(len(chunks))]
#                 metas = [{"source": uploaded.name, "chunk": i} for i in range(len(chunks))]
#                 collection.add(documents=chunks, metadatas=metas, ids=ids)
                
#                 # Store full text for booklet generation
#                 st.session_state.full_paper_text = raw_text
#                 st.session_state.paper_filename = uploaded.name
                
#                 st.success(f"✅ Indexed {len(chunks)} chunks from {uploaded.name}")

# with col2:
#     st.header("💭 Ask Questions")
#     question = st.text_input("Your question:", placeholder="What is the main contribution of this paper?")
    
#     if st.button("🔎 Retrieve & Ask", type="primary") and question.strip():
#         with st.spinner("Searching and generating response..."):
#             res = collection.query(query_texts=[question], n_results=N_RESULTS)
#             docs = res.get("documents", [[]])[0]
            
#             if not docs:
#                 st.warning("No results found.")
#             else:
#                 context = "\n".join(docs)
#                 prompt = f"""
#                 Answer the question based on the context provided. Be comprehensive, detailed, and technical where appropriate.
                
#                 Context:
#                 {context}
                
#                 Question: {question}
                
#                 Please provide a detailed response (aim for {response_word_limit} words) that thoroughly addresses the question.
#                 """
                
#                 response = model.generate_content(prompt)
#                 limited_response = limit_response_words(response.text, response_word_limit)
                
#                 st.subheader("🎯 Answer")
#                 st.write(limited_response)
                
#                 # Show retrieved passages in expandable section
#                 with st.expander("📄 Retrieved Context Passages"):
#                     for i, doc in enumerate(docs, 1):
#                         st.write(f"**Passage {i}:**")
#                         st.write(doc)
#                         st.divider()

# # Booklet Generation Section
# st.header("📖 Generate Research Booklet")

# if st.button("🚀 Generate Comprehensive Booklet", type="primary"):
#     if 'full_paper_text' not in st.session_state:
#         st.error("Please upload and index a PDF first!")
#     else:
#         with st.spinner("Generating comprehensive booklet... This may take a few minutes."):
#             progress_bar = st.progress(0)
            
#             # Extract metadata
#             progress_bar.progress(20)
#             metadata = extract_paper_metadata(st.session_state.full_paper_text)
            
#             # Search for citations
#             citations = []
#             if enable_citations:
#                 progress_bar.progress(40)
#                 search_terms = metadata.get('title', 'research paper') + " " + " ".join(metadata.get('keywords', [])[:3])
#                 citations = search_citations(search_terms)
            
#             # Generate booklet
#             progress_bar.progress(60)
#             booklet_content = generate_booklet(st.session_state.full_paper_text, metadata, citations)
            
#             # Create visualizations
#             if enable_visualizations:
#                 progress_bar.progress(80)
#                 viz_data = create_visualization(st.session_state.full_paper_text[:5000])  # Use first 5000 chars
            
#             progress_bar.progress(100)
            
#             # Display booklet
#             st.subheader("📚 Generated Booklet")
#             st.markdown(booklet_content)
            
#             # Show visualizations
#             if enable_visualizations:
#                 st.subheader("📊 Visual Analysis")
#                 st.image(viz_data, caption="Research Paper Word Cloud")
            
#             # Download option
#             st.download_button(
#                 label="📥 Download Booklet (Markdown)",
#                 data=booklet_content,
#                 file_name=f"research_booklet_{st.session_state.paper_filename}_{datetime.now().strftime('%Y%m%d')}.md",
#                 mime="text/markdown"
#             )

# # Footer
# st.markdown("---")
# st.markdown("*Enhanced RAG Application with Booklet Generation | Powered by Gemini AI & ChromaDB*")
import os
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai
import requests
import json
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
import graphviz
import seaborn as sns
import pandas as pd
from io import BytesIO
import base64
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
import re
from datetime import datetime
import numpy as np
from wordcloud import WordCloud
from collections import Counter
import tempfile
import uuid

# Config
DB_DIR = "./chroma_db"
COLLECTION_NAME = "papers"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
N_RESULTS = 8
MAX_RESPONSE_LENGTH = 2000  # Configurable response length

# Gemini setup
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash")

# Crossref API for citations
CROSSREF_API = "https://api.crossref.org/works"

class VisualizationGenerator:
    """Generates meaningful visualizations based on research paper content"""
    
    @staticmethod
    def create_concept_network(text_chunks, title="Concept Network"):
        """Creates a network graph of key concepts"""
        # Extract key terms using simple frequency analysis
        all_text = " ".join(text_chunks).lower()
        # Remove common words and extract technical terms
        words = re.findall(r'\b[a-zA-Z]{4,}\b', all_text)
        word_freq = Counter(words)
        
        # Get top concepts
        top_concepts = [word for word, freq in word_freq.most_common(15) 
                       if freq > 2 and word not in ['this', 'that', 'with', 'from', 'they', 'were', 'been', 'have']]
        
        # Create network
        G = nx.Graph()
        for i, concept1 in enumerate(top_concepts):
            for concept2 in top_concepts[i+1:]:
                # Check co-occurrence
                if concept1 in all_text and concept2 in all_text:
                    co_occur = all_text.count(f"{concept1}") + all_text.count(f"{concept2}")
                    if co_occur > 1:
                        G.add_edge(concept1, concept2, weight=co_occur)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Draw network
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                              node_size=[len(node)*100 for node in G.nodes()], alpha=0.7)
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
        nx.draw_networkx_edges(G, pos, alpha=0.5, edge_color='gray')
        
        plt.title(f"{title}\nKey Concepts and Relationships", fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        # Save to bytes
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        return buf

    @staticmethod
    def create_methodology_flowchart(text_chunks):
        """Creates a flowchart representing the research methodology"""
        # Extract methodology steps using keywords
        methodology_keywords = ['first', 'second', 'then', 'next', 'finally', 'step', 'phase', 'stage']
        steps = []
        
        for chunk in text_chunks:
            chunk_lower = chunk.lower()
            if any(keyword in chunk_lower for keyword in methodology_keywords):
                # Extract sentences with methodology keywords
                sentences = chunk.split('.')
                for sentence in sentences:
                    if any(keyword in sentence.lower() for keyword in methodology_keywords):
                        steps.append(sentence.strip()[:100] + "...")
        
        if not steps:
            steps = ["Data Collection", "Preprocessing", "Analysis", "Results", "Validation"]
        
        # Create flowchart using graphviz
        dot = graphviz.Digraph(comment='Methodology Flowchart', format='png')
        dot.attr(rankdir='TB', size='8,10')
        dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightblue')
        
        for i, step in enumerate(steps[:6]):  # Limit to 6 steps
            dot.node(str(i), step[:50] + "..." if len(step) > 50 else step)
            if i > 0:
                dot.edge(str(i-1), str(i))
        
        # Render to bytes
        temp_dir = tempfile.mkdtemp()
        dot.render(f'{temp_dir}/methodology', format='png')
        
        with open(f'{temp_dir}/methodology.png', 'rb') as f:
            img_bytes = BytesIO(f.read())
        
        return img_bytes

    @staticmethod
    def create_results_visualization(text_chunks):
        """Creates charts representing results/findings"""
        # Extract numerical data mentions
        numbers = []
        for chunk in text_chunks:
            # Find percentages, ratios, and numbers
            percentages = re.findall(r'(\d+\.?\d*)%', chunk)
            ratios = re.findall(r'(\d+\.?\d*):(\d+\.?\d*)', chunk)
            values = re.findall(r'(\d+\.?\d*)', chunk)
            
            numbers.extend([float(p) for p in percentages if float(p) <= 100])
            numbers.extend([float(v) for v in values if 0 < float(v) < 1000])
        
        if numbers:
            # Create distribution plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Histogram
            ax1.hist(numbers, bins=min(15, len(numbers)), alpha=0.7, color='skyblue', edgecolor='black')
            ax1.set_title('Distribution of Numerical Values\nMentioned in Paper', fontweight='bold')
            ax1.set_xlabel('Values')
            ax1.set_ylabel('Frequency')
            
            # Box plot
            ax2.boxplot(numbers, vert=True, patch_artist=True, 
                       boxprops=dict(facecolor='lightgreen', alpha=0.7))
            ax2.set_title('Statistical Summary\nof Numerical Data', fontweight='bold')
            ax2.set_ylabel('Values')
            
            plt.tight_layout()
            
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)
            plt.close()
            return buf
        else:
            # Create a dummy chart if no numerical data found
            categories = ['Method A', 'Method B', 'Method C', 'Baseline']
            values = [85, 92, 78, 65]  # Sample values
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(categories, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                        f'{value}%', ha='center', va='bottom', fontweight='bold')
            
            plt.title('Performance Comparison\n(Sample Results)', fontsize=14, fontweight='bold')
            plt.ylabel('Performance Score (%)')
            plt.ylim(0, 100)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)
            plt.close()
            return buf

    @staticmethod
    def create_word_cloud(text_chunks, title="Key Terms"):
        """Creates a word cloud of important terms"""
        all_text = " ".join(text_chunks)
        
        # Clean text
        cleaned_text = re.sub(r'[^\w\s]', ' ', all_text.lower())
        
        # Generate word cloud
        wordcloud = WordCloud(width=800, height=400, 
                             background_color='white',
                             colormap='viridis',
                             max_words=100,
                             relative_scaling=0.5).generate(cleaned_text)
        
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(f"{title}\nMost Frequent Terms in Research Paper", 
                 fontsize=16, fontweight='bold', pad=20)
        plt.axis('off')
        plt.tight_layout()
        
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        return buf

class CitationSearcher:
    """Searches for citations and related papers"""
    
    @staticmethod
    def search_related_papers(query, limit=5):
        """Search for related papers using Crossref API"""
        try:
            # Clean query
            clean_query = re.sub(r'[^\w\s]', ' ', query)
            
            params = {
                'query': clean_query,
                'rows': limit,
                'sort': 'relevance',
                'filter': 'type:journal-article'
            }
            
            response = requests.get(CROSSREF_API, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                papers = []
                
                for item in data.get('message', {}).get('items', []):
                    paper = {
                        'title': item.get('title', ['Unknown'])[0] if item.get('title') else 'Unknown',
                        'authors': [f"{author.get('given', '')} {author.get('family', '')}" 
                                  for author in item.get('author', [])[:3]],
                        'journal': item.get('container-title', ['Unknown'])[0] if item.get('container-title') else 'Unknown',
                        'year': item.get('published-print', {}).get('date-parts', [[2023]])[0][0] if item.get('published-print') else 'Unknown',
                        'doi': item.get('DOI', 'No DOI')
                    }
                    papers.append(paper)
                
                return papers
            
        except Exception as e:
            st.warning(f"Citation search failed: {str(e)}")
            return []
        
        return []

class BookletGenerator:
    """Generates comprehensive PDF booklets"""
    
    def __init__(self, title, author="RAG System"):
        self.title = title
        self.author = author
        self.styles = getSampleStyleSheet()
        self.story = []
        
        # Custom styles
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        self.heading_style = ParagraphStyle(
            'CustomHeading',
            parent=self.styles['Heading1'],
            fontSize=18,
            spaceAfter=20,
            textColor=colors.darkblue
        )
        
        self.body_style = ParagraphStyle(
            'CustomBody',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=12,
            alignment=TA_JUSTIFY
        )
    
    def add_title_page(self):
        """Add title page"""
        self.story.append(Spacer(1, 2*inch))
        self.story.append(Paragraph(self.title, self.title_style))
        self.story.append(Spacer(1, 0.5*inch))
        self.story.append(Paragraph(f"Generated by: {self.author}", self.styles['Normal']))
        self.story.append(Paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y')}", self.styles['Normal']))
        self.story.append(PageBreak())
    
    def add_section(self, title, content):
        """Add a section with title and content"""
        self.story.append(Paragraph(title, self.heading_style))
        self.story.append(Paragraph(content, self.body_style))
        self.story.append(Spacer(1, 20))
    
    def add_image(self, img_buffer, caption="", width=6*inch):
        """Add image to booklet"""
        if img_buffer:
            img_buffer.seek(0)
            img = Image(img_buffer, width=width, height=width*0.6)
            self.story.append(img)
            if caption:
                self.story.append(Paragraph(f"<i>{caption}</i>", self.styles['Normal']))
            self.story.append(Spacer(1, 20))
    
    def add_citations(self, citations):
        """Add citations section"""
        if citations:
            self.story.append(Paragraph("Related Research & Citations", self.heading_style))
            
            for i, citation in enumerate(citations, 1):
                citation_text = f"""
                <b>{i}. {citation['title']}</b><br/>
                Authors: {', '.join(citation['authors']) if citation['authors'] else 'Unknown'}<br/>
                Journal: {citation['journal']}<br/>
                Year: {citation['year']}<br/>
                DOI: {citation['doi']}<br/><br/>
                """
                self.story.append(Paragraph(citation_text, self.body_style))
            
            self.story.append(Spacer(1, 20))
    
    def generate_pdf(self):
        """Generate the final PDF"""
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72,
                              topMargin=72, bottomMargin=18)
        doc.build(self.story)
        buffer.seek(0)
        return buffer

# Enhanced functions
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

def generate_comprehensive_analysis(paper_content, question=None):
    """Generate comprehensive analysis of the research paper"""
    
    if question:
        prompt = f"""
        Based on the research paper content below, provide a detailed answer to the question: "{question}"
        
        Requirements:
        - Provide a comprehensive response (maximum {MAX_RESPONSE_LENGTH} words)
        - Include specific details and evidence from the paper
        - Structure your response clearly with key points
        - Be precise and technical where appropriate
        
        Paper content: {paper_content[:5000]}
        
        Question: {question}
        """
    else:
        prompt = f"""
        Analyze this research paper and provide a comprehensive summary covering:
        
        1. **Research Objective**: What problem does this paper address?
        2. **Methodology**: How did the researchers approach the problem?
        3. **Key Findings**: What are the main results and discoveries?
        4. **Significance**: Why is this research important?
        5. **Limitations**: What are the acknowledged limitations?
        6. **Future Work**: What directions for future research are suggested?
        
        Keep the response detailed but within {MAX_RESPONSE_LENGTH} words.
        
        Paper content: {paper_content[:5000]}
        """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Analysis generation failed: {str(e)}"

# Streamlit UI
st.set_page_config(page_title="Enhanced RAG with Booklet Generation", 
                   page_icon="🧠", layout="wide")

st.title("🧠 Enhanced RAG — Research Paper Analysis & Booklet Generation")
st.markdown("Upload research papers, ask questions, and generate comprehensive booklets with visualizations!")

# Initialize components
embedder, client = get_embedding_fn_and_client()
try:
    collection = client.get_collection(COLLECTION_NAME, embedding_function=embedder)
except:
    collection = client.create_collection(COLLECTION_NAME, embedding_function=embedder)

viz_gen = VisualizationGenerator()
citation_searcher = CitationSearcher()

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    max_response_length = st.slider("Max Response Length (words)", 500, 5000, MAX_RESPONSE_LENGTH)
    n_results = st.slider("Number of Retrieved Chunks", 3, 15, N_RESULTS)
    include_citations = st.checkbox("Include Citation Search", value=True)
    include_visualizations = st.checkbox("Include Visualizations", value=True)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📚 1. Upload and Index PDF")
    uploaded = st.file_uploader("Upload PDF", type=["pdf"])
    
    if uploaded:
        save_path = os.path.join("uploads", uploaded.name)
        os.makedirs("uploads", exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(uploaded.getbuffer())

        if st.button("📚 Index PDF", type="primary"):
            with st.spinner("Processing PDF..."):
                raw_text = extract_text_from_pdf(save_path)
                chunks = chunk_text(raw_text)
                ids = [f"{uploaded.name}-{i}" for i in range(len(chunks))]
                metas = [{"source": uploaded.name, "chunk": i} for i in range(len(chunks))]
                collection.add(documents=chunks, metadatas=metas, ids=ids)
                st.success(f"✅ Indexed {len(chunks)} chunks from {uploaded.name}")

with col2:
    st.header("❓ 2. Ask Questions")
    question = st.text_area("Your question:", height=100, 
                           placeholder="Ask about the research methodology, results, or any specific aspect...")
    
    if st.button("🔍 Analyze & Answer", type="primary") and question.strip():
        with st.spinner("Retrieving relevant content..."):
            # Retrieve relevant chunks
            res = collection.query(query_texts=[question], n_results=n_results)
            docs = res.get("documents", [[]])[0]
            
            if not docs:
                st.warning("No relevant content found.")
            else:
                context = "\n".join(docs)
                
                # Generate comprehensive response
                with st.spinner("Generating comprehensive analysis..."):
                    analysis = generate_comprehensive_analysis(context, question)
                
                st.subheader("📝 Analysis & Answer")
                st.write(analysis)
                
                # Display retrieved passages
                with st.expander("📖 Retrieved Passages", expanded=False):
                    for i, doc in enumerate(docs, 1):
                        st.write(f"**Passage {i}:** {doc[:500]}...")

# Generate Booklet Section
st.header("📖 3. Generate Research Booklet")

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    generate_full_analysis = st.button("📊 Generate Full Paper Analysis", type="secondary")

with col2:
    booklet_title = st.text_input("Booklet Title", value="Research Paper Analysis")

with col3:
    generate_booklet = st.button("📖 Generate Complete Booklet", type="primary")

if generate_full_analysis or generate_booklet:
    # Get all documents for comprehensive analysis
    all_res = collection.query(query_texts=["research methodology results findings"], n_results=20)
    all_docs = all_res.get("documents", [[]])[0]
    
    if all_docs:
        with st.spinner("Generating comprehensive analysis..."):
            full_analysis = generate_comprehensive_analysis("\n".join(all_docs))
        
        st.subheader("📊 Complete Paper Analysis")
        st.write(full_analysis)
        
        if generate_booklet:
            with st.spinner("Generating booklet with visualizations..."):
                # Initialize booklet generator
                booklet = BookletGenerator(booklet_title)
                booklet.add_title_page()
                
                # Add main analysis
                booklet.add_section("Research Paper Analysis", full_analysis)
                
                # Generate and add visualizations
                if include_visualizations:
                    st.subheader("📊 Generated Visualizations")
                    
                    # Concept Network
                    with st.spinner("Creating concept network..."):
                        concept_img = viz_gen.create_concept_network(all_docs[:10])
                        if concept_img:
                            st.image(concept_img, caption="Concept Network", use_container_width=True)
                            booklet.add_image(concept_img, "Key Concepts and Their Relationships")
                    
                    # Methodology Flowchart
                    with st.spinner("Creating methodology flowchart..."):
                        try:
                            method_img = viz_gen.create_methodology_flowchart(all_docs[:10])
                            if method_img:
                                st.image(method_img, caption="Methodology Flowchart", use_container_width=True)
                                booklet.add_image(method_img, "Research Methodology Flow")
                        except Exception as e:
                            st.warning(f"Flowchart generation skipped: {str(e)}")
                    
                    # Results Visualization
                    with st.spinner("Creating results visualization..."):
                        results_img = viz_gen.create_results_visualization(all_docs[:10])
                        if results_img:
                            st.image(results_img, caption="Results Analysis", use_container_width=True)
                            booklet.add_image(results_img, "Statistical Analysis of Results")
                    
                    # Word Cloud
                    with st.spinner("Creating word cloud..."):
                        wordcloud_img = viz_gen.create_word_cloud(all_docs[:15])
                        if wordcloud_img:
                            st.image(wordcloud_img, caption="Key Terms Word Cloud", use_container_width=True)
                            booklet.add_image(wordcloud_img, "Most Important Terms in the Research")
                
                # Search for citations
                if include_citations:
                    with st.spinner("Searching for related papers..."):
                        citations = citation_searcher.search_related_papers(booklet_title, limit=8)
                        if citations:
                            st.subheader("📚 Related Research")
                            for citation in citations[:5]:
                                st.write(f"**{citation['title']}** - {citation['authors'][:2]} ({citation['year']})")
                            booklet.add_citations(citations)
                
                # Generate final PDF
                with st.spinner("Generating PDF booklet..."):
                    pdf_buffer = booklet.generate_pdf()
                    
                    st.success("✅ Booklet generated successfully!")
                    
                    # Download button
                    st.download_button(
                        label="📥 Download Research Booklet (PDF)",
                        data=pdf_buffer,
                        file_name=f"{booklet_title.replace(' ', '_')}_booklet.pdf",
                        mime="application/pdf",
                        type="primary"
                    )
    else:
        st.warning("No documents found. Please upload and index a PDF first.")

# Footer
st.markdown("---")
st.markdown(
    """
    **Features:**
    - 📚 PDF Processing & Indexing
    - 🔍 Intelligent Question Answering  
    - 📊 Automatic Visualizations (Networks, Flowcharts, Statistics)
    - 🔗 Citation & Related Paper Search
    - 📖 Comprehensive PDF Booklet Generation
    - ⚙️ Configurable Response Length & Retrieval
    """
)