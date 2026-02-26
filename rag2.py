import os
import warnings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Prefer non-deprecated packages; fall back to community to avoid extra deps
try:
    from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore[import-untyped]
except ImportError:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        from langchain_community.embeddings import HuggingFaceEmbeddings

try:
    from langchain_ollama import ChatOllama  # type: ignore[import-untyped]
except ImportError:
    from langchain_community.chat_models import ChatOllama


def _check_ollama_reachable(base_url: str = "http://localhost:11434", timeout: int = 5) -> None:
    """Raise a clear error if Ollama is not running."""
    try:
        import urllib.request
        req = urllib.request.Request(f"{base_url.rstrip('/')}/api/tags", method="GET")
        urllib.request.urlopen(req, timeout=timeout)
    except OSError as e:
        raise ConnectionError(
            "Ollama is not running or not reachable. Start it with: ollama serve (or open the Ollama app)."
        ) from e


class RAGPipeline:
    def __init__(self, use_ollama=True, ollama_model="llama3.2", google_api_key=None):
        """
        Initializes the RAG Pipeline.
        
        Args:
           use_ollama: If True, use local Ollama (no API key). If False, use Google Gemini.
           ollama_model: Ollama model name (e.g. llama3.2, llama2, mistral). Used when use_ollama=True.
           google_api_key: Google Gemini API key. Required only when use_ollama=False.
        """
        self.use_ollama = use_ollama
        self.ollama_model = ollama_model
        self.google_api_key = google_api_key
        # Use a lightweight local embedding model for low latency
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
            self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = None
        self.chain = None

    def ingest_data(self, scraped_data):
        """
        Processes scraped data, chunks it, and builds the FAISS vector store.
        
        Args:
            scraped_data: list of dicts with 'url' and 'content' keys.
        """
        documents = []
        for item in scraped_data:
            doc = Document(
                page_content=item['content'],
                metadata={"source": item['url']}
            )
            documents.append(doc)
            
        # Chunking: split documents into smaller chunks for precise retrieval
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        splits = text_splitter.split_documents(documents)
        
        if not splits:
            raise ValueError("No text could be extracted from the provided URL(s).")
            
        # Create In-memory vector store
        self.vector_store = FAISS.from_documents(splits, self.embeddings)
        
        # Setup the LLM and Chain
        self._setup_chain()

    def _setup_chain(self):
        """Builds the conversational retrieval chain."""
        if self.use_ollama:
            # Local Ollama - no API key needed (requires Ollama installed and model pulled)
            _check_ollama_reachable()
            # base_url: explicit default; timeout: avoid failures on slow RAG responses
            try:
                llm = ChatOllama(
                    model=self.ollama_model,
                    base_url="http://localhost:11434",
                    temperature=0,
                    timeout=120,
                )
            except TypeError:
                # Some ChatOllama versions don't accept base_url/timeout
                llm = ChatOllama(model=self.ollama_model, temperature=0)
        else:
            # Google Gemini - API key required
            from langchain_google_genai import ChatGoogleGenerativeAI
            if not self.google_api_key:
                raise ValueError("Google API key is required when not using Ollama.")
            os.environ["GOOGLE_API_KEY"] = self.google_api_key
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

        assert self.vector_store is not None  # set in ingest_data before _setup_chain
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})

        # Prompt to contextualize the user's question with chat history
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
        
        # Prompt for answering the question based on context
        qa_system_prompt = (
            "You are a helpful and knowledgeable assistant for a website.\n"
            "Use the following pieces of retrieved context to answer the user's question accurately.\n"
            "If you don't know the answer based on the context, just say that you don't know, "
            "don't try to make up an answer.\n"
            "Keep the answer concise but fully address the user's prompt.\n"
            "Cite your sources if relevant based on the metadata.\n"
            "\nContext:\n{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        self.chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    def ask(self, question, chat_history=None):
        """
        Ask a question to the RAG pipeline.

        Args:
            question: The user's query string.
            chat_history: List of past messages for context (list of HumanMessage/AIMessage).

        Returns:
            The generated answer string.
        """
        if chat_history is None:
            chat_history = []
        if not self.chain:
            raise ValueError("Data has not been ingested yet. Call ingest_data first.")

        try:
            response = self.chain.invoke({"input": question, "chat_history": chat_history})
        except Exception as e:
            err_msg = str(e).lower()
            if "connection" in err_msg or "refused" in err_msg or "connect" in err_msg:
                raise ConnectionError(
                    "Cannot reach Ollama. Is it running? Start the Ollama app or run 'ollama serve' in a terminal."
                ) from e
            if "not found" in err_msg or "404" in err_msg:
                raise FileNotFoundError(
                    f"Ollama model '{self.ollama_model}' not found. Run: ollama pull {self.ollama_model}"
                ) from e
            raise

        answer = response.get("answer") if isinstance(response, dict) else None
        if answer is None:
            raise ValueError(f"Model returned no answer. Response keys: {list(response.keys()) if isinstance(response, dict) else 'not a dict'}")
        if not isinstance(answer, str):
            answer = str(answer)
        return answer.strip() or "I couldn't generate an answer. Try rephrasing or check if Ollama is running."