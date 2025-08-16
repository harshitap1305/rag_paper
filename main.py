import streamlit as st
import os

# Import all modules
from config import COLLECTION_NAME, N_RESULTS, MAX_RESPONSE_LENGTH
from pdf_processing import extract_text_from_pdf, chunk_text
from database_manager import get_embedding_fn_and_client, clear_collection_and_create_new, get_current_paper_docs
from analysis_generator import generate_comprehensive_analysis
from visualizations import AdvancedVisualizationGenerator
from citation_search import CitationSearcher
from booklet_generator import BookletGenerator
from utils import display_image

# Initialize visualization generator and citation searcher
viz_gen = AdvancedVisualizationGenerator()
citation_searcher = CitationSearcher()

# Streamlit UI
st.set_page_config(page_title="Enhanced RAG with Professional Visualizations", 
                   page_icon="🧠", layout="wide")

st.title("🧠 Enhanced RAG — Research Paper Analysis & Professional Booklet Generation")
st.markdown("Upload research papers, ask questions, and generate comprehensive booklets with **sophisticated visualizations**!")

# Initialize session state
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
    st.header("⚙️ Settings")
    max_response_length = st.slider("Max Response Length (words)", 500, 5000, MAX_RESPONSE_LENGTH)
    n_results = st.slider("Number of Retrieved Chunks", 3, 15, N_RESULTS)
    include_citations = st.checkbox("Include Citation Search", value=True)
    include_visualizations = st.checkbox("Include Visualizations", value=True)
    
    # Visualization options (removed concept network and wordcloud)
    st.subheader("📊 Visualization Types")
    viz_methodology_flow = st.checkbox("Methodology Flowchart", value=True)
    viz_results_dashboard = st.checkbox("Results Dashboard", value=True)
    viz_timeline = st.checkbox("Research Timeline", value=True)
    viz_hierarchy = st.checkbox("Concept Hierarchy", value=True)
    
    # Current paper info
    st.header("📄 Current Paper")
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
        if st.session_state.current_paper != uploaded.name:
            st.session_state.paper_indexed = False
        
        save_path = os.path.join("uploads", uploaded.name)
        os.makedirs("uploads", exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(uploaded.getbuffer())

        if st.button("📚 Index PDF", type="primary"):
            with st.spinner("Processing PDF..."):
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
                        
                        with st.spinner("Generating comprehensive analysis..."):
                            analysis = generate_comprehensive_analysis(context, question, max_response_length)
                        
                        st.subheader("📝 Analysis & Answer")
                        st.write(analysis)
                        
                        with st.expander("📖 Retrieved Passages", expanded=False):
                            for i, doc in enumerate(docs, 1):
                                st.write(f"**Passage {i}:** {doc[:500]}...")
            except Exception as e:
                st.error(f"Error during query: {str(e)}")

# Generate Booklet Section
st.header("📖 3. Generate Professional Research Booklet")

if not st.session_state.paper_indexed:
    st.info("Please upload and index a PDF first")
else:
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        generate_full_analysis = st.button("📊 Generate Full Paper Analysis", type="secondary")

    with col2:
        booklet_title = st.text_input("Booklet Title", value=f"Analysis of {st.session_state.current_paper}" if st.session_state.current_paper else "Research Paper Analysis")

    with col3:
        generate_booklet = st.button("📖 Generate Professional Booklet", type="primary")

    if generate_full_analysis or generate_booklet:
        try:
            collection = client.get_collection(COLLECTION_NAME, embedding_function=embedder)
            all_docs = get_current_paper_docs(collection, st.session_state.current_paper)
            
            if all_docs:
                with st.spinner("Generating comprehensive analysis..."):
                    full_analysis = generate_comprehensive_analysis("\n".join(all_docs), max_length=max_response_length)
                
                st.subheader(f"📊 Complete Analysis of {st.session_state.current_paper}")
                st.write(full_analysis)
                
                if generate_booklet:
                    with st.spinner("Generating professional booklet with visualizations..."):
                        booklet = BookletGenerator(booklet_title)
                        booklet.add_title_page()
                        booklet.add_section("Research Paper Analysis", full_analysis)
                        
                        if include_visualizations:
                            st.subheader("📊 Visualizations")
                            viz_docs = all_docs[:15] if len(all_docs) > 15 else all_docs
                            
                            # Methodology Flowchart
                            if viz_methodology_flow:
                                with st.spinner("Creating methodology flow..."):
                                    try:
                                        method_img = viz_gen.create_methodology_flow(viz_docs[:10])
                                        if method_img:
                                            display_image(method_img, caption="Research Methodology Flowchart")
                                            booklet.add_image(method_img, "Research Methodology Workflow")
                                    except Exception as e:
                                        st.warning(f"Could not create methodology flowchart: {str(e)}")
                            
                            # Results Dashboard
                            if viz_results_dashboard:
                                with st.spinner("Creating results dashboard..."):
                                    try:
                                        results_img = viz_gen.create_advanced_results_dashboard(viz_docs[:10])
                                        if results_img:
                                            display_image(results_img, caption="Results Dashboard")
                                            booklet.add_image(results_img, "Comprehensive Results Analysis Dashboard")
                                    except Exception as e:
                                        st.warning(f"Could not create results dashboard: {str(e)}")
                            
                            # Research Timeline
                            if viz_timeline:
                                with st.spinner("Creating research timeline..."):
                                    try:
                                        timeline_img = viz_gen.create_research_timeline(viz_docs[:10])
                                        if timeline_img:
                                            display_image(timeline_img, caption="Research Project Timeline")
                                            booklet.add_image(timeline_img, "Research Timeline Visualization")
                                    except Exception as e:
                                        st.warning(f"Could not create timeline: {str(e)}")
                            
                            # Concept Hierarchy
                            if viz_hierarchy:
                                with st.spinner("Creating concept hierarchy..."):
                                    try:
                                        hierarchy_img = viz_gen.create_concept_hierarchy(viz_docs[:15])
                                        if hierarchy_img:
                                            display_image(hierarchy_img, caption="Concept Hierarchy Diagram")
                                            booklet.add_image(hierarchy_img, "Concept Structure Analysis")
                                    except Exception as e:
                                        st.warning(f"Could not create concept hierarchy: {str(e)}")
                        
                        # Search for citations with links
                        if include_citations:
                            with st.spinner("Searching for related papers..."):
                                try:
                                    search_query = booklet_title if booklet_title else st.session_state.current_paper
                                    citations = citation_searcher.search_related_papers(search_query, limit=8)
                                    if citations:
                                        st.subheader("📚 Related Research")
                                        for citation in citations[:5]:
                                            authors_str = ', '.join(citation['authors'][:2]) if citation['authors'] else 'Unknown'
                                            if citation.get('url'):
                                                st.write(f"**[{citation['title']}]({citation['url']})** - {authors_str} ({citation['year']})")
                                            else:
                                                st.write(f"**{citation['title']}** - {authors_str} ({citation['year']})")
                                        booklet.add_citations(citations)
                                except Exception as e:
                                    st.warning(f"Could not search for citations: {str(e)}")
                        
                        # Generate final PDF
                        with st.spinner("Generating professional PDF booklet..."):
                            pdf_buffer = booklet.generate_pdf()
                            
                            if pdf_buffer:
                                st.success("✅ Professional booklet generated successfully!")
                                
                                st.download_button(
                                    label="📥 Download Professional Research Booklet (PDF)",
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
    **🌟 Enhanced Features:**
    - 📚 **Advanced PDF Processing** & Intelligent Indexing
    - 🔍 **Smart Question Answering** with Context Retrieval
    - 🎨 **Professional Visualizations**: 
        - 📊 Advanced Results Dashboards  
        - 🔄 Methodology Flowcharts
        - 📅 Research Timeline Diagrams
        - 🏗️ Hierarchical Concept Maps
    - 🔗 **Citation Search with Clickable Links**
    - 📖 **Professional PDF Booklet Generation** with Enhanced Formatting
    - ⚙️ **Configurable Settings** & Analysis Parameters
    - 🗑️ **Smart Database Management** (Single Paper Focus)
    
    **🎯 Professional Quality:** All visualizations and PDFs are formatted for academic presentations and reports.
    """
)