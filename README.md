# RAG-Powered Website Chatbot

A conversational chatbot that scrapes websites and answers questions using Retrieval-Augmented Generation (RAG). Supports both local Ollama models (free, no API key) and Google Gemini.

## Features

- **Web Scraping**: Recursively scrapes websites with configurable depth and page limits
- **Dual LLM Support**: 
  - **Ollama** - Local, free, no API key required
  - **Google Gemini** - Cloud-based, requires API key
- **Conversational RAG**: Maintains chat history for context-aware responses
- **Fast Retrieval**: Uses FAISS vector store with sentence-transformer embeddings
- **Modern UI**: Dark-themed Streamlit interface with responsive design

## Quick Start

### Prerequisites

- Python 3.9+
- For Ollama: [Install Ollama](https://ollama.com) (Windows/Mac/Linux)

### Installation

```bash
# Clone repository
git clone <repo-url>
cd RAG-Powered-Website-Chatbot

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

### Option 1: Use Ollama (Free, Local, No API Key)

```bash
# Pull a model (one-time, ~2GB)
ollama pull llama3.2

# Run the app
streamlit run app.py
```

In the sidebar, select **"Ollama (local, free)"**, enter a website URL, and click **"Scrape & Initialize Model"**.

### Option 2: Use Google Gemini

```bash
# Copy example env file
cp .env.example .env

# Edit .env and add your Google API key
echo 'GOOGLE_API_KEY="your-key-here"' > .env

# Run the app
streamlit run app.py
```

In the sidebar, select **"Google Gemini (API Key)"**, enter your URL, and click **"Scrape & Initialize Model"**.

## Project Structure

| File | Description |
|------|-------------|
| `app.py` | Streamlit web interface |
| `scraper.py` | Recursive web scraper with BeautifulSoup |
| `rag2.py` | RAG pipeline (Ollama/Gemini + FAISS + LangChain) |
| `requirements.txt` | Python dependencies |
| `.env.example` | Template for environment variables |
| `run.bat` | Windows launcher script |

## How It Works

1. **Scraping**: `RecursiveWebScraper` extracts content from URLs recursively (max depth: 2, max pages: 20)
2. **Processing**: `RAGPipeline` chunks documents and creates FAISS vector index
3. **Retrieval**: User queries trigger semantic search over indexed content
4. **Generation**: LLM generates answers using retrieved context

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Ollama is not running" | Start Ollama with `ollama serve` or open the app |
| "Connection timed out" | Check internet connection |
| Import errors | Delete `__pycache__` folders and restart |

## License

MIT License