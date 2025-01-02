import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize session state
if "initialized" not in st.session_state:
    st.session_state.initialized = False

def init_groq():
    """Initialize Groq API key and model"""
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("❌ GROQ_API_KEY not found in environment variables!")
        return None
    
    try:
        return ChatGroq(
            groq_api_key=groq_api_key,
            model_name="gemma2-9b-it",
            temperature=0.5,
            max_tokens=1024
        )
    except Exception as e:
        st.error(f"Error initializing Groq: {str(e)}")
        return None

def create_vector_embedding():
    """Create vector embeddings for documents"""
    try:
        with st.spinner("Please Wait"):
            # Simplified embeddings initialization with corrected parameters
            st.session_state.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={
                    'normalize_embeddings': True,
                    'batch_size': 32
                }
            )
            
            # Check if documents directory exists
            if not os.path.exists("draft_ruu_kuhp_final"):
                os.makedirs("draft_ruu_kuhp_final")
                st.warning("📁 Created 'draft_ruu_kuhp_final' directory. Please add your PDF files there.")
                return False
            
            # Load and process documents
            st.session_state.loader = PyPDFDirectoryLoader("draft_ruu_kuhp_final")
            st.session_state.docs = st.session_state.loader.load()
            
            if not st.session_state.docs:
                st.error("❌ No documents found in the directory!")
                return False
            
            # Split documents into chunks
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2500,
                chunk_overlap=100
            )
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(
                st.session_state.docs
            )
            
            # Create vector store
            st.session_state.vectors = FAISS.from_documents(
                st.session_state.final_documents,
                st.session_state.embeddings
            )
            
            st.session_state.initialized = True
            return True
            
    except Exception as e:
        st.error(f"Error during embedding creation: {str(e)}")
        return False

def load_existing_vectorstore():
    """Load existing vector store if available"""
    try:
        if os.path.exists("faiss_index"):
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            st.session_state.vectors = FAISS.load_local("faiss_index", embeddings)
            st.session_state.initialized = True
            return True
    except Exception as e:
        st.error(f"Error loading existing vector store: {str(e)}")
    return False

def main():
    # Page configuration
    st.set_page_config(
        page_title="🔴 ChatBot: Indonesian Criminal Law",
        page_icon="⚖️",
        layout="wide"
    )

    # Check if the image exists
    image_path = "image1.JPG"
    if os.path.exists(image_path):
        st.image(image_path, use_container_width=True)
    else:
        st.warning("⚠️ Header image not found. Please ensure 'image1.JPG' is in the correct directory.")
    
    # Title with red styling
    st.markdown(
        "<h1 style='text-align: center; color: #B22222;'>ChatBot: Hukum Pidana (Criminal Law)</h1>",
        unsafe_allow_html=True
    )
    
    # Add sidebar with instructions
    with st.sidebar:
        st.header("Instructions")
        st.markdown("""
        1. Main document used is "draft_ruu_kuhp_final"
        2. Click 'Start Consultation'
        3. Ask your questions related to Indonesian Criminal Law
        """)
    
    # Initialize Groq
    llm = init_groq()
    if not llm:
        st.stop()
    
    # Try to load existing vector store
    if not st.session_state.initialized:
        if load_existing_vectorstore():
            st.success("✅ Loaded existing vector database!")
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_template("""
    You are a highly knowledgeable legal expert and AI assistant specializing in Indonesian criminal law (KUHP). 
        Your role is to:
        - Answer legal questions about the KUHP professionally and wisely.
        - Discuss cases and suggest applicable 'pasal' (articles) with precise legal interpretations.
        - Provide clear and concise legal guidance, with accurate references to the KUHP.
        - Provide a step-by-step guide and suggest possible solutions for resolving the legal case based on the relevant laws.

        Language Guidelines:
        - Always respond in the user's preferred language. If the user does not specify, respond in formal Indonesian.
        - Use professional and respectful language, avoiding casual expressions.
        - For complex legal terms, provide explanations in simple terms without losing the formality.

    <context>
    {context}
    </context>

    Question: {input}
                                            
    Based on the context and query, provide a wise, professional, and multilingual legal response. 
    Include the relevant 'pasal' references when appropriate and explain them if needed.
    """)
    
    # Document embedding button
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Start Consultation"):
            if create_vector_embedding():
                st.success("✅ What can I help with?")
            else:
                st.error("❌ Document Error.")
    
    # User input
    user_prompt = st.text_input("Ask your questions related to Indonesian Criminal Law:", 
                               placeholder="e.g., saya jadi korban mafia tanah, apa yang harus saya lakukan?")
    
    # Process query
    if user_prompt:
        if not st.session_state.initialized:
            st.warning("⚠️ Please initialize document embedding first!")
            return
            
        try:
            with st.spinner("Processing your query..."):
                # Create chains
                document_chain = create_stuff_documents_chain(llm, prompt)
                retriever = st.session_state.vectors.as_retriever(
                    search_kwargs={"k": 10}  # Retrieve top 3 most relevant documents
                )
                retrieval_chain = create_retrieval_chain(retriever, document_chain)
                
                # Get response
                start = time.process_time()
                response = retrieval_chain.invoke({'input': user_prompt})
                response_time = time.process_time() - start
                
                # Display results
                st.write(f"⏱️ Response time: {response_time:.2f} seconds")
                
                st.markdown("### Answer:")
                st.write(response['answer'])
                
                # Show similar documents
                with st.expander("📑 Excerpts from the Draft Criminal Code (RKHUP)"):
                    for i, doc in enumerate(response['context']):
                        st.markdown(f"**Document {i+1}:**")
                        st.write(doc.page_content)
                        if hasattr(doc, 'metadata') and doc.metadata:
                            st.caption(f"Source: {doc.metadata.get('source', 'Unknown')}")
                        st.markdown("---")
                        
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")

if __name__ == "__main__":
    main()
