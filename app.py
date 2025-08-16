import os
import warnings
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

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Set matplotlib backend to avoid display issues
import matplotlib
matplotlib.use('Agg')

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

# Helper function to handle Streamlit version compatibility
def display_image(image_data, caption="", width=None):
    """Display image with proper Streamlit compatibility"""
    try:
        # Try newer parameter first
        st.image(image_data, caption=caption, use_container_width=True)
    except TypeError:
        try:
            # Try older parameter
            st.image(image_data, caption=caption, use_column_width=True)
        except TypeError:
            # Fallback to basic display
            st.image(image_data, caption=caption, width=width or 700)

class VisualizationGenerator:
    """Generates meaningful visualizations based on research paper content"""
    
    @staticmethod
    def create_concept_network(text_chunks, title="Concept Network"):
        """Creates a network graph of key concepts"""
        try:
            # Extract key terms using simple frequency analysis
            all_text = " ".join(text_chunks).lower()
            # Remove common words and extract technical terms
            words = re.findall(r'\b[a-zA-Z]{4,}\b', all_text)
            word_freq = Counter(words)
            
            # Get top concepts
            top_concepts = [word for word, freq in word_freq.most_common(15) 
                           if freq > 2 and word not in ['this', 'that', 'with', 'from', 'they', 'were', 'been', 'have']]
            
            if len(top_concepts) < 3:
                top_concepts = ['research', 'method', 'analysis', 'data', 'results']
            
            # Create network
            G = nx.Graph()
            for i, concept1 in enumerate(top_concepts[:10]):  # Limit to 10 concepts
                for concept2 in top_concepts[i+1:]:
                    # Check co-occurrence
                    if concept1 in all_text and concept2 in all_text:
                        co_occur = all_text.count(f"{concept1}") + all_text.count(f"{concept2}")
                        if co_occur > 1:
                            G.add_edge(concept1, concept2, weight=co_occur)
            
            # Ensure we have some nodes even if no edges
            if len(G.nodes()) == 0:
                for concept in top_concepts[:8]:
                    G.add_node(concept)
            
            # Create visualization
            plt.figure(figsize=(12, 8))
            
            if len(G.nodes()) > 0:
                pos = nx.spring_layout(G, k=3, iterations=50)
                
                # Draw network
                nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                                      node_size=[len(node)*100 for node in G.nodes()], alpha=0.7)
                nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
                if len(G.edges()) > 0:
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
        except Exception as e:
            st.warning(f"Could not generate concept network: {str(e)}")
            return None

    @staticmethod
    def create_methodology_flowchart(text_chunks):
        """Creates a flowchart representing the research methodology"""
        try:
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
                            clean_sentence = sentence.strip()
                            if len(clean_sentence) > 10:
                                steps.append(clean_sentence[:100] + "..." if len(clean_sentence) > 100 else clean_sentence)
            
            if not steps:
                steps = ["Data Collection", "Data Preprocessing", "Model Development", "Analysis & Evaluation", "Results Validation"]
            
            # Create simple matplotlib flowchart instead of graphviz
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Limit steps to avoid overcrowding
            steps = steps[:6]
            
            # Create boxes for each step
            box_height = 0.8
            box_width = 1.5
            y_positions = np.linspace(len(steps), 1, len(steps))
            
            for i, (step, y_pos) in enumerate(zip(steps, y_positions)):
                # Draw box
                box = plt.Rectangle((0.5, y_pos - box_height/2), box_width, box_height, 
                                  facecolor='lightblue', edgecolor='black', linewidth=2)
                ax.add_patch(box)
                
                # Add text
                text = step[:40] + "..." if len(step) > 40 else step
                ax.text(0.5 + box_width/2, y_pos, text, 
                       ha='center', va='center', fontsize=10, wrap=True)
                
                # Draw arrow to next step
                if i < len(steps) - 1:
                    ax.arrow(0.5 + box_width/2, y_pos - box_height/2 - 0.1, 
                            0, -0.3, head_width=0.1, head_length=0.1, 
                            fc='black', ec='black')
            
            ax.set_xlim(0, 2.5)
            ax.set_ylim(0, len(steps) + 1)
            ax.set_title('Research Methodology Flowchart', fontsize=16, fontweight='bold')
            ax.axis('off')
            plt.tight_layout()
            
            # Save to bytes
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)
            plt.close()
            return buf
            
        except Exception as e:
            st.warning(f"Could not generate methodology flowchart: {str(e)}")
            return None

    @staticmethod
    def create_results_visualization(text_chunks):
        """Creates charts representing results/findings"""
        try:
            # Extract numerical data mentions
            numbers = []
            for chunk in text_chunks:
                # Find percentages, ratios, and numbers
                percentages = re.findall(r'(\d+\.?\d*)%', chunk)
                values = re.findall(r'(\d+\.?\d*)', chunk)
                
                numbers.extend([float(p) for p in percentages if 0 <= float(p) <= 100])
                numbers.extend([float(v) for v in values if 0 < float(v) < 1000])
            
            if len(numbers) > 3:
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
                # Create a sample chart if no numerical data found
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
        except Exception as e:
            st.warning(f"Could not generate results visualization: {str(e)}")
            return None

    @staticmethod
    def create_word_cloud(text_chunks, title="Key Terms"):
        """Creates a word cloud of important terms"""
        try:
            all_text = " ".join(text_chunks)
            
            # Clean text
            cleaned_text = re.sub(r'[^\w\s]', ' ', all_text.lower())
            
            # Remove common stop words
            stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'this', 'that', 'these', 'those', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
            words = cleaned_text.split()
            filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
            cleaned_text = ' '.join(filtered_words)
            
            if len(cleaned_text.strip()) == 0:
                cleaned_text = "research analysis method data results findings"
            
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
        except Exception as e:
            st.warning(f"Could not generate word cloud: {str(e)}")
            return None

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
            try:
                img_buffer.seek(0)
                img = Image(img_buffer, width=width, height=width*0.6)
                self.story.append(img)
                if caption:
                    self.story.append(Paragraph(f"<i>{caption}</i>", self.styles['Normal']))
                self.story.append(Spacer(1, 20))
            except Exception as e:
                st.warning(f"Could not add image to booklet: {str(e)}")
    
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
        try:
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72,
                                  topMargin=72, bottomMargin=18)
            doc.build(self.story)
            buffer.seek(0)
            return buffer
        except Exception as e:
            st.error(f"Could not generate PDF: {str(e)}")
            return None

# Enhanced functions
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
        # Try to delete existing collection
        try:
            client.delete_collection(COLLECTION_NAME)
        except:
            pass  # Collection might not exist
        
        # Create new collection
        collection = client.create_collection(COLLECTION_NAME, embedding_function=embedder)
        return collection
    except Exception as e:
        st.error(f"Error creating new collection: {str(e)}")
        return None

def get_current_paper_docs(collection, current_paper_name):
    """Get documents only from the current paper"""
    try:
        # Query with a filter for the current paper
        all_docs = collection.get(
            where={"source": current_paper_name}
        )
        return all_docs.get("documents", [])
    except Exception as e:
        st.error(f"Error retrieving current paper documents: {str(e)}")
        return []

def generate_comprehensive_analysis(paper_content, question=None, max_length=MAX_RESPONSE_LENGTH):
    """Generate comprehensive analysis of the research paper"""
    
    if question:
        prompt = f"""
        Based on the research paper content below, provide a detailed answer to the question: "{question}"
        
        Requirements:
        - Provide a comprehensive response (maximum {max_length} words)
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
        
        Keep the response detailed but within {max_length} words.
        
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

# Initialize session state for current paper tracking
if 'current_paper' not in st.session_state:
    st.session_state.current_paper = None
if 'paper_indexed' not in st.session_state:
    st.session_state.paper_indexed = False

# Initialize components
embedder, client = get_embedding_fn_and_client()

if embedder is None or client is None:
    st.error("Failed to initialize embedding function or client. Please check your setup.")
    st.stop()

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    max_response_length = st.slider("Max Response Length (words)", 500, 5000, MAX_RESPONSE_LENGTH)
    n_results = st.slider("Number of Retrieved Chunks", 3, 15, N_RESULTS)
    include_citations = st.checkbox("Include Citation Search", value=True)
    include_visualizations = st.checkbox("Include Visualizations", value=True)
    
    # Current paper info
    st.header("Current Paper")
    if st.session_state.current_paper:
        st.success(f"📄 {st.session_state.current_paper}")
    else:
        st.info("No paper loaded")
    
    # Clear database button
    if st.button("🗑️ Clear Database", type="secondary"):
        try:
            client.delete_collection(COLLECTION_NAME)
            st.session_state.current_paper = None
            st.session_state.paper_indexed = False
            st.success("Database cleared!")
            st.rerun()
        except:
            st.info("Database was already empty")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📚 1. Upload and Index PDF")
    uploaded = st.file_uploader("Upload PDF", type=["pdf"])
    
    if uploaded:
        # Check if this is a new paper
        if st.session_state.current_paper != uploaded.name:
            st.session_state.paper_indexed = False
        
        save_path = os.path.join("uploads", uploaded.name)
        os.makedirs("uploads", exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(uploaded.getbuffer())

        if st.button("📚 Index PDF", type="primary"):
            with st.spinner("Processing PDF..."):
                # Clear existing collection and create new one for this paper
                collection = clear_collection_and_create_new(client, embedder)
                
                if collection is not None:
                    raw_text = extract_text_from_pdf(save_path)
                    if raw_text:
                        chunks = chunk_text(raw_text)
                        if chunks:
                            ids = [f"{uploaded.name}-{i}" for i in range(len(chunks))]
                            metas = [{"source": uploaded.name, "chunk": i} for i in range(len(chunks))]
                            try:
                                collection.add(documents=chunks, metadatas=metas, ids=ids)
                                st.session_state.current_paper = uploaded.name
                                st.session_state.paper_indexed = True
                                st.success(f"✅ Indexed {len(chunks)} chunks from {uploaded.name}")
                                st.success(f"🔄 Database cleared and loaded with current paper only")
                            except Exception as e:
                                st.error(f"Error indexing PDF: {str(e)}")
                        else:
                            st.error("No text chunks extracted from PDF")
                    else:
                        st.error("No text extracted from PDF")

with col2:
    st.header("❓ 2. Ask Questions")
    
    if not st.session_state.paper_indexed:
        st.info("Please upload and index a PDF first")
    else:
        question = st.text_area("Your question:", height=100, 
                               placeholder="Ask about the research methodology, results, or any specific aspect...")
        
        if st.button("🔍 Analyze & Answer", type="primary") and question.strip():
            try:
                collection = client.get_collection(COLLECTION_NAME, embedding_function=embedder)
                
                with st.spinner("Retrieving relevant content..."):
                    # Retrieve relevant chunks from current paper only
                    res = collection.query(
                        query_texts=[question], 
                        n_results=n_results,
                        where={"source": st.session_state.current_paper}
                    )
                    docs = res.get("documents", [[]])[0]
                    
                    if not docs:
                        st.warning("No relevant content found in the current paper.")
                    else:
                        context = "\n".join(docs)
                        
                        # Generate comprehensive response
                        with st.spinner("Generating comprehensive analysis..."):
                            analysis = generate_comprehensive_analysis(context, question, max_response_length)
                        
                        st.subheader("📝 Analysis & Answer")
                        st.write(analysis)
                        
                        # Display retrieved passages
                        with st.expander("📖 Retrieved Passages", expanded=False):
                            for i, doc in enumerate(docs, 1):
                                st.write(f"**Passage {i}:** {doc[:500]}...")
            except Exception as e:
                st.error(f"Error during query: {str(e)}")

# Generate Booklet Section
st.header("📖 3. Generate Research Booklet")

if not st.session_state.paper_indexed:
    st.info("Please upload and index a PDF first")
else:
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        generate_full_analysis = st.button("📊 Generate Full Paper Analysis", type="secondary")

    with col2:
        booklet_title = st.text_input("Booklet Title", value=f"Analysis of {st.session_state.current_paper}" if st.session_state.current_paper else "Research Paper Analysis")

    with col3:
        generate_booklet = st.button("📖 Generate Complete Booklet", type="primary")

    if generate_full_analysis or generate_booklet:
        try:
            collection = client.get_collection(COLLECTION_NAME, embedding_function=embedder)
            
            # Get documents from current paper only
            all_docs = get_current_paper_docs(collection, st.session_state.current_paper)
            
            if all_docs:
                with st.spinner("Generating comprehensive analysis..."):
                    full_analysis = generate_comprehensive_analysis("\n".join(all_docs), max_length=max_response_length)
                
                st.subheader(f"📊 Complete Analysis of {st.session_state.current_paper}")
                st.write(full_analysis)
                
                if generate_booklet:
                    with st.spinner("Generating booklet with visualizations..."):
                        # Initialize booklet generator
                        booklet = BookletGenerator(booklet_title)
                        booklet.add_title_page()
                        
                        # Add main analysis
                        booklet.add_section("Research Paper Analysis", full_analysis)
                        
                        # Generate and add visualizations
                        if include_visualizations and viz_gen is not None:
                            st.subheader("📊 Generated Visualizations")
                            
                            # Use only current paper's documents for visualizations
                            viz_docs = all_docs[:15] if len(all_docs) > 15 else all_docs
                            
                            # Concept Network
                            with st.spinner("Creating concept network..."):
                                try:
                                    concept_img = viz_gen.create_concept_network(viz_docs[:10], st.session_state.current_paper)
                                    if concept_img:
                                        display_image(concept_img, caption="Concept Network")
                                        booklet.add_image(concept_img, "Key Concepts and Their Relationships")
                                except Exception as e:
                                    st.warning(f"Could not create concept network: {str(e)}")
                            
                            # Methodology Flowchart
                            with st.spinner("Creating methodology flowchart..."):
                                try:
                                    method_img = viz_gen.create_methodology_flowchart(viz_docs[:10])
                                    if method_img:
                                        display_image(method_img, caption="Methodology Flowchart")
                                        booklet.add_image(method_img, "Research Methodology Flow")
                                except Exception as e:
                                    st.warning(f"Could not create methodology flowchart: {str(e)}")
                            
                            # Results Visualization
                            with st.spinner("Creating results visualization..."):
                                try:
                                    results_img = viz_gen.create_results_visualization(viz_docs[:10])
                                    if results_img:
                                        display_image(results_img, caption="Results Analysis")
                                        booklet.add_image(results_img, "Statistical Analysis of Results")
                                except Exception as e:
                                    st.warning(f"Could not create results visualization: {str(e)}")
                            
                            # Word Cloud
                            with st.spinner("Creating word cloud..."):
                                try:
                                    wordcloud_img = viz_gen.create_word_cloud(viz_docs, f"Key Terms - {st.session_state.current_paper}")
                                    if wordcloud_img:
                                        display_image(wordcloud_img, caption="Key Terms Word Cloud")
                                        booklet.add_image(wordcloud_img, "Most Important Terms in the Research")
                                except Exception as e:
                                    st.warning(f"Could not create word cloud: {str(e)}")
                        elif include_visualizations:
                            st.warning("Visualization generator not available. Skipping visualizations.")
                        
                        # Search for citations
                        if include_citations and citation_searcher is not None:
                            with st.spinner("Searching for related papers..."):
                                try:
                                    # Use current paper name for citation search
                                    search_query = booklet_title if booklet_title else st.session_state.current_paper
                                    citations = citation_searcher.search_related_papers(search_query, limit=8)
                                    if citations:
                                        st.subheader("📚 Related Research")
                                        for citation in citations[:5]:
                                            authors_str = ', '.join(citation['authors'][:2]) if citation['authors'] else 'Unknown'
                                            st.write(f"**{citation['title']}** - {authors_str} ({citation['year']})")
                                        booklet.add_citations(citations)
                                except Exception as e:
                                    st.warning(f"Could not search for citations: {str(e)}")
                        elif include_citations:
                            st.warning("Citation searcher not available. Skipping citation search.")
                        
                        # Generate final PDF
                        with st.spinner("Generating PDF booklet..."):
                            pdf_buffer = booklet.generate_pdf()
                            
                            if pdf_buffer:
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
                                st.error("Failed to generate PDF booklet")
            else:
                st.warning(f"No documents found for {st.session_state.current_paper}. Please reindex the PDF.")
        except Exception as e:
            st.error(f"Error during booklet generation: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    """
    **Features:**
    - 📚 PDF Processing & Indexing (Single Paper Mode)
    - 🔍 Intelligent Question Answering  
    - 📊 Automatic Visualizations (Networks, Flowcharts, Statistics)
    - 🔗 Citation & Related Paper Search
    - 📖 Comprehensive PDF Booklet Generation
    - ⚙️ Configurable Response Length & Retrieval
    - 🗑️ Database Management (Clear & Reset)
    
    **Note:** Each new PDF upload clears the database and loads only the current paper to ensure isolated analysis.
    """
)