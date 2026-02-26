import streamlit as st
import logging
import traceback
from langchain_core.messages import HumanMessage, AIMessage

from scraper import RecursiveWebScraper
from rag2 import RAGPipeline

# Basic page config for modern look
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")

# Inject some custom CSS for styling
st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .main-header {
        text-align: center;
        margin-bottom: 2rem;
        background: -webkit-linear-gradient(45deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>Conversational RAG Chatbot</h1>", unsafe_allow_html=True)

# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = None
if "is_ingested" not in st.session_state:
    st.session_state.is_ingested = False

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    llm_choice = st.radio(
        "Choose LLM (no API key needed for Ollama)",
        options=["Ollama (local, free)", "Google Gemini (API key required)"],
        index=0,
        help="Ollama runs models on your PC. Install from ollama.com and run e.g. 'ollama pull llama3.2'",
    )
    use_ollama = llm_choice.startswith("Ollama")
    
    if use_ollama:
        ollama_model = st.text_input("Ollama model name", value="llama3.2", help="e.g. llama3.2, llama2, mistral. Run 'ollama list' to see installed models.")
        api_key = None
        st.caption("Ensure Ollama is running (Ollama app or run: ollama serve)")
    else:
        ollama_model = "llama3.2"
        api_key = st.text_input("Google Gemini API Key", type="password")
    
    start_url = st.text_input("Website URL to Scrape", placeholder="https://example.com")
    max_depth = st.slider("Scraping Depth", min_value=1, max_value=3, value=1, help="Depth 1 = just this page. Depth 2 = this page and all links on it.")
    max_pages = st.number_input("Max Pages Limit", min_value=1, max_value=50, value=10)
    
    if st.button("Scrape & Initialize Model", type="primary"):
        if not start_url:
            st.error("Please provide a valid URL.")
        elif not use_ollama and not api_key:
            st.error("Please provide a Google Gemini API Key, or switch to Ollama (local, free).")
        else:
            with st.spinner(f"Scraping up to {max_pages} pages from {start_url}..."):
                try:
                    # 1. Scrape
                    scraper = RecursiveWebScraper(start_url, max_depth=max_depth, max_pages=max_pages)
                    scraped_data = scraper.get_data()
                    
                    if not scraped_data:
                        st.error("Could not extract any content from the given URL.")
                    else:
                        st.success(f"Successfully scraped {len(scraped_data)} pages.")
                        
                        # 2. Ingest
                        with st.spinner("Processing documents into vector store (Minimal latency via FAISS)..."):
                            pipeline = RAGPipeline(
                                use_ollama=use_ollama,
                                ollama_model=ollama_model.strip() or "llama3.2",
                                google_api_key=api_key,
                            )
                            pipeline.ingest_data(scraped_data)
                            
                            st.session_state.rag_pipeline = pipeline
                            st.session_state.is_ingested = True
                            st.session_state.chat_history = [] # Reset chat on new ingestion
                            
                        st.success("Ready! You can now ask questions.")
                except Exception as e:
                    st.error(f"**An error occurred:** {e}")
                    logging.exception("Scrape/ingest failed: %s", e)
                    st.markdown("**Full error (traceback):**")
                    st.code(traceback.format_exc(), language="text")

# Main Chat Interface
if st.session_state.is_ingested:
    # Display chat messages from history on app rerun
    for msg in st.session_state.chat_history:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)

    # Accept user input
    if prompt := st.chat_input("Ask a question about the website..."):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    pipeline = st.session_state.rag_pipeline
                    if pipeline is None:
                        st.error("Pipeline not initialized. Please scrape and initialize first.")
                    else:
                        answer = pipeline.ask(prompt, st.session_state.chat_history)
                        st.markdown(answer)
                        # Update chat history
                        st.session_state.chat_history.append(HumanMessage(content=prompt))
                        st.session_state.chat_history.append(AIMessage(content=answer))
                except Exception as e:
                    err_text = str(e)
                    st.error(f"**Error generating answer:** {err_text}")
                    if "Ollama" in err_text or "ollama" in err_text.lower():
                        st.warning("Tip: Make sure Ollama is running (open the Ollama app or run `ollama serve` in a terminal).")
                    st.markdown("**Full error (traceback):**")
                    st.code(traceback.format_exc(), language="text")
else:
    st.info("👈 Choose **Ollama (local, free)** in the sidebar, enter a URL, then click 'Scrape & Initialize Model' to begin. No API key needed.")