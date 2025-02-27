import streamlit as st
import os
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Ollama
import time
import random
import pandas as pd

# Set page configuration with a custom theme and favicon
st.set_page_config(
    page_title="PDF Insight Engine",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        font-weight: 800;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        font-weight: 600;
    }
    .success-box {
        background-color: #EEFBEE;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #4CAF50;
    }
    .info-box {
        background-color: #EFF8FB;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #2196F3;
    }
    .error-box {
        background-color: #FEEFEF;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #F44336;
    }
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #0D47A1;
    }
    .chat-message-user {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .chat-message-ai {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 3px solid #1E88E5;
    }
    .pdf-list-item {
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin-bottom: 0.3rem;
        border: 1px solid #EEEEEE;
    }
    .sidebar-header {
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_pdfs" not in st.session_state:
    st.session_state.processed_pdfs = []
if "extracted_context" not in st.session_state:
    st.session_state.extracted_context = ""
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.1
if "chunk_size" not in st.session_state:
    st.session_state.chunk_size = 1000
if "chunk_overlap" not in st.session_state:
    st.session_state.chunk_overlap = 200
if "model_name" not in st.session_state:
    st.session_state.model_name = "llama3"
if "progress" not in st.session_state:
    st.session_state.progress = 0
if "current_pdf" not in st.session_state:
    st.session_state.current_pdf = ""
if "chat_initialized" not in st.session_state:
    st.session_state.chat_initialized = False

def initialize_llm():
    model_name = st.session_state.model_name
    
    llm = Ollama(
        model=model_name,
        temperature=st.session_state.temperature,
        num_ctx=4096  # Context window size
    )
    return llm

def simulate_progress(total_pdfs):
    """Simulate progress for a better UX when processing PDFs"""
    progress_bar = st.progress(0)
    step_size = 1.0 / (total_pdfs * 10)
    
    for pdf_index in range(total_pdfs):
        st.session_state.current_pdf = st.session_state.processed_pdfs[pdf_index]
        for i in range(10):
            # Add a slight delay and randomness for realism
            time.sleep(0.1 + random.uniform(0, 0.2))
            st.session_state.progress += step_size
            progress_bar.progress(min(st.session_state.progress, 1.0))
            
    # Complete the progress
    progress_bar.progress(1.0)
    time.sleep(0.5)
    return progress_bar

def process_pdfs(pdf_docs):
    total_new_pdfs = len([pdf for pdf in pdf_docs if pdf.name not in st.session_state.processed_pdfs])
    
    if total_new_pdfs == 0:
        st.warning("All uploaded PDFs have already been processed.")
        return
        
    st.session_state.progress = 0
    progress_container = st.empty()
    status_text = st.empty()
    
    with progress_container:
        progress_bar = st.progress(0)
    
    # Create a temporary directory to store uploaded PDFs
    with tempfile.TemporaryDirectory() as temp_dir:
        text_chunks = []
        all_text = []
        
        for i, pdf in enumerate(pdf_docs):
            if pdf.name in st.session_state.processed_pdfs:
                continue
                
            st.session_state.current_pdf = pdf.name
            status_text.text(f"Processing: {pdf.name} ({i+1}/{total_new_pdfs})")
            
            # Save the uploaded PDF to the temporary directory
            file_path = os.path.join(temp_dir, pdf.name)
            with open(file_path, "wb") as f:
                f.write(pdf.getbuffer())
            
            # Load and process the PDF
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            # Collect all text for context extraction
            pdf_text = "\n\n".join([doc.page_content for doc in documents])
            all_text.append(f"--- Content from {pdf.name} ---\n{pdf_text}")
            
            # Split the documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=st.session_state.chunk_size,
                chunk_overlap=st.session_state.chunk_overlap,
                length_function=len
            )
            chunks = text_splitter.split_documents(documents)
            text_chunks.extend(chunks)
            
            # Add to processed PDFs list
            st.session_state.processed_pdfs.append(pdf.name)
            
            # Update progress
            st.session_state.progress = (i + 1) / total_new_pdfs
            progress_bar.progress(st.session_state.progress)
        
        # Store all extracted text as context
        st.session_state.extracted_context = "\n\n".join(all_text)
        
        # Create embeddings and vector store
        status_text.text("Creating embeddings and knowledge base...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        if text_chunks:
            if hasattr(st.session_state, 'vectorstore'):
                # Add to existing vectorstore
                st.session_state.vectorstore.add_documents(text_chunks)
            else:
                # Create new vectorstore
                st.session_state.vectorstore = FAISS.from_documents(text_chunks, embeddings)
    
    # Create conversation chain
    if hasattr(st.session_state, 'vectorstore'):
        status_text.text("Initializing conversation chain...")
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        st.session_state.conversation = ConversationalRetrievalChain.from_llm(
            llm=initialize_llm(),
            retriever=st.session_state.vectorstore.as_retriever(),
            memory=memory
        )
        st.session_state.chat_initialized = True
    
    progress_bar.progress(1.0)
    status_text.empty()
    progress_container.empty()

def get_response(user_query):
    if st.session_state.conversation is None:
        return "⚠️ Please upload and process PDFs first to initialize the system."
    
    response = st.session_state.conversation({"question": user_query})
    return response["answer"]

def get_pdf_stats():
    """Calculate and return stats about the processed PDFs"""
    if not hasattr(st.session_state, 'vectorstore'):
        return None
    
    stats = {
        "total_pdfs": len(st.session_state.processed_pdfs),
        "total_chunks": len(st.session_state.vectorstore.index_to_docstore_id),
    }
    
    return stats

def display_pdf_list():
    """Display the list of processed PDFs with nice formatting"""
    if not st.session_state.processed_pdfs:
        st.info("No PDFs have been processed yet.")
        return
    
    for i, pdf in enumerate(st.session_state.processed_pdfs, 1):
        st.markdown(f"""
        <div class="pdf-list-item">
            <strong>{i}.</strong> {pdf}
        </div>
        """, unsafe_allow_html=True)

def main():
    st.markdown('<p class="main-header">📚 PDF Insight Engine</p>', unsafe_allow_html=True)
    st.markdown("Extract knowledge and insights from your PDF documents using local LLMs")
    
    # Sidebar for model configuration
    with st.sidebar:
        st.markdown('<p class="sidebar-header">🤖 LLM Configuration</p>', unsafe_allow_html=True)
        
        # Check if Ollama is running
        import requests
        try:
            response = requests.get("http://localhost:11434/api/tags")
            ollama_running = response.status_code == 200
            available_models = [model["name"] for model in response.json()["models"]] if ollama_running else []
        except:
            ollama_running = False
            available_models = []
        
        if not ollama_running:
            st.markdown("""
            <div class="error-box">
                ⚠️ Local LLM server is not running or not accessible.
                <br/>Make sure Ollama is running on localhost:11434.
            </div>
            """, unsafe_allow_html=True)
            model_name = st.text_input("Enter model name manually:", value="llama3")
            st.session_state.model_name = model_name
        else:
            st.markdown("""
            <div class="success-box">
                ✅ Local LLM server is running!
            </div>
            """, unsafe_allow_html=True)
            
            if available_models:
                model_name = st.selectbox("Select a model", available_models)
                st.session_state.model_name = model_name
            else:
                st.warning("No models found. Please pull a model using the Ollama app.")
                st.code("Example: Run 'ollama pull llama3' from terminal or use the Ollama app")
                model_name = st.text_input("Or enter model name manually:", value="llama3")
                st.session_state.model_name = model_name
        
        st.markdown('<p class="sidebar-header">📄 Upload Documents</p>', unsafe_allow_html=True)
        pdf_docs = st.file_uploader("Upload your PDFs", type=["pdf"], accept_multiple_files=True)
        
        col1, col2 = st.columns(2)
        with col1:
            process_button = st.button("Process PDFs", use_container_width=True)
        with col2:
            if st.button("Clear All", use_container_width=True):
                # Clear session state
                for key in ["conversation", "chat_history", "processed_pdfs", "extracted_context"]:
                    if key in st.session_state:
                        st.session_state[key] = [] if key == "chat_history" or key == "processed_pdfs" else ""
                if "vectorstore" in st.session_state:
                    del st.session_state.vectorstore
                st.session_state.chat_initialized = False
                st.rerun()
        
        # Process PDFs when button is clicked
        if process_button:
            if not pdf_docs:
                st.error("Please upload at least one PDF")
            else:
                with st.spinner("Processing PDFs and extracting context..."):
                    process_pdfs(pdf_docs)
                st.success("PDFs processed successfully!")
        
        # Display PDF statistics if available
        stats = get_pdf_stats()
        if stats:
            st.markdown('<p class="sidebar-header">📊 Knowledge Base Stats</p>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="info-box">
                📚 <strong>Total PDFs:</strong> {stats["total_pdfs"]}<br/>
                🧩 <strong>Total Chunks:</strong> {stats["total_chunks"]}<br/>
                🔍 <strong>Model:</strong> {st.session_state.model_name}<br/>
                🌡️ <strong>Temperature:</strong> {st.session_state.temperature}
            </div>
            """, unsafe_allow_html=True)
        
        # Show processed PDFs in sidebar
        if st.session_state.processed_pdfs:
            st.markdown('<p class="sidebar-header">📋 Processed Documents</p>', unsafe_allow_html=True)
            display_pdf_list()
    
    # Main area with tabs
    tab1, tab2, tab3 = st.tabs(["📝 Extracted Context", "💬 Chat with Your PDFs", "⚙️ Settings"])
    
    with tab1:
        st.markdown('<p class="sub-header">📝 Extracted Context from PDFs</p>', unsafe_allow_html=True)
        if st.session_state.extracted_context:
            st.text_area("Extracted content:", value=st.session_state.extracted_context, height=400, disabled=True)
            
            # Add option to download the extracted context
            col1, col2 = st.columns([3, 1])
            with col2:
                st.download_button(
                    label="⬇️ Download Context",
                    data=st.session_state.extracted_context,
                    file_name="extracted_context.txt",
                    mime="text/plain",
                    use_container_width=True
                )
        else:
            st.info("Upload and process PDFs to extract context")
    
    with tab2:
        st.markdown('<p class="sub-header">💬 Chat with Your PDFs</p>', unsafe_allow_html=True)
        
        # Chat interface
        if not st.session_state.chat_initialized:
            st.markdown("""
            <div class="info-box">
                👋 Welcome to the PDF Insight Engine chat!
                <br/><br/>
                To get started:
                <ol>
                    <li>Upload PDF documents via the sidebar</li>
                    <li>Click "Process PDFs" to extract knowledge</li>
                    <li>Ask questions about your documents here</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Display chat container
            chat_container = st.container()
            
            # Input area for user question
            with st.container():
                user_query = st.text_input("Ask a question about your PDFs:", key="user_question", placeholder="e.g., What are the main points mentioned in these documents?")
                
                # Clear chat and Ask buttons in a row
                col1, col2 = st.columns([4, 1])
                with col2:
                    if st.button("Clear Chat", use_container_width=True):
                        st.session_state.chat_history = []
                        st.rerun()
                
                if user_query:
                    if not hasattr(st.session_state, 'conversation') or st.session_state.conversation is None:
                        st.error("Please process PDFs first before asking questions")
                    else:
                        with st.spinner("Analyzing your documents..."):
                            response = get_response(user_query)
                        
                        # Add to chat history
                        st.session_state.chat_history.append({"user": user_query, "bot": response})
                        
                        # Clear input using st.query_params instead of the deprecated function
                        st.query_params.clear()
            
            # Display chat history
            with chat_container:
                if not st.session_state.chat_history:
                    st.info("Your conversation will appear here.")
                else:
                    for message in st.session_state.chat_history:
                        st.markdown(f"""
                        <div class="chat-message-user">
                            <strong>You:</strong> {message['user']}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown(f"""
                        <div class="chat-message-ai">
                            <strong>AI:</strong> {message['bot']}
                        </div>
                        """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<p class="sub-header">⚙️ Settings</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### LLM Parameters")
            temperature = st.slider(
                "Temperature (randomness)", 
                min_value=0.0, 
                max_value=1.0, 
                value=st.session_state.temperature, 
                step=0.1,
                help="Higher values make the output more random, lower values make it more focused and deterministic."
            )
        
        with col2:
            st.markdown("#### Text Processing")
            chunk_size = st.number_input(
                "Chunk Size", 
                min_value=100, 
                max_value=8000, 
                value=st.session_state.chunk_size, 
                step=100,
                help="The size of text chunks for processing. Larger chunks provide more context but may be less precise."
            )
            chunk_overlap = st.number_input(
                "Chunk Overlap", 
                min_value=0, 
                max_value=500, 
                value=st.session_state.chunk_overlap, 
                step=50,
                help="The amount of text that overlaps between chunks to maintain context continuity."
            )
        
        # Advanced settings expandable section
        with st.expander("Advanced Settings"):
            st.markdown("#### Retrieval Settings")
            k_value = st.slider(
                "Number of retrieval results", 
                min_value=1, 
                max_value=10, 
                value=4, 
                help="The number of document chunks to retrieve for each query"
            )
            
            st.markdown("#### Performance Settings")
            num_threads = st.slider(
                "Number of threads", 
                min_value=1, 
                max_value=8, 
                value=4, 
                help="Number of CPU threads to use for processing"
            )
        
        # Apply settings button
        if st.button("Apply Settings", use_container_width=True):
            # Store settings in session state
            st.session_state.temperature = temperature
            st.session_state.chunk_size = chunk_size
            st.session_state.chunk_overlap = chunk_overlap
            
            # Re-initialize conversation with new settings if it exists
            if hasattr(st.session_state, 'vectorstore'):
                memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
                st.session_state.conversation = ConversationalRetrievalChain.from_llm(
                    llm=initialize_llm(),
                    retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": k_value}),
                    memory=memory
                )
            
            st.success("Settings updated successfully!")

if __name__ == "__main__":
    main()





