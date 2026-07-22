# Multi-Utility-Chatbot

An agentic chatbot built with LangGraph that answers general questions, searches the web, performs calculations, and answers questions grounded in your own PDF documents using Retrieval-Augmented Generation (RAG).

Unlike a plain PDF Q&A bot, this is a multi-tool agent — the LLM dynamically decides whether to answer from its own knowledge, search the web, run a calculation, or retrieve from an uploaded document.

Features
💬 General chat — answers any question directly, no document required
📄 PDF RAG — attach a PDF in the chat and ask questions grounded in its content
🔍 Web search — fetches current information via DuckDuckGo
🧮 Calculator & stock prices — tool calling for math and market data
🗂️ Persistent conversations — chat history saved in SQLite, survives restarts
🏷️ Auto-generated titles — LLM names each conversation from the first message
🗑️ Delete conversations — remove threads from memory, ChatGPT-style
🔄 Provider-agnostic — runs fully local with Ollama, or with Gemini/OpenAI APIs

Architecture
<img width="826" height="381" alt="image" src="https://github.com/user-attachments/assets/01f53475-3ad9-49c6-928d-396e64a2cf3a" />



RAG pipeline: PDF → PyPDFLoader → RecursiveCharacterTextSplitter (1000 chars, 200 overlap) → embeddings → FAISS vector store → similarity retrieval (top-k=4) → context injected into the LLM prompt.

Tech Stack
Layer	Technology
Orchestration	LangGraph (StateGraph, ToolNode, conditional edges)
LLM	Ollama (llama3.2) / Google Gemini / OpenAI
Embeddings	nomic-embed-text / gemini-embedding-001
Vector store	FAISS
Persistence	SQLite (LangGraph SqliteSaver)
Frontend	Streamlit
PDF parsing	PyPDF
Setup
1. Clone and install
bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
2. Install Ollama (for the local, free setup)

Download from ollama.com, then pull both required models:

bash
ollama pull llama3.2
ollama pull nomic-embed-text

Verify they're installed:

bash
ollama list
3. Run
bash
streamlit run streamlit_rag_frontend.py

Open http://localhost:8501 in your browser.

Usage
Ask anything — type a question and get an answer from the model's knowledge
Attach a PDF — click the attach icon in the chat bar, upload, then ask about it
Switch conversations — click any title in the sidebar
Delete a conversation — click the 🗑 next to it
Using an API model instead of Ollama

To use Google Gemini (or OpenAI) instead of local models, replace the LLM setup in langraph_rag_backend.py:

python
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

Add your key to a .env file:

GOOGLE_API_KEY=your_key_here

Note: switching embedding providers changes vector dimensions. Delete chatbot.db and re-upload your PDFs after switching.

Project Structure
.
├── langraph_rag_backend.py       # LangGraph agent, tools, RAG pipeline, persistence
├── streamlit_rag_frontend.py     # Streamlit chat UI
├── requirements.txt
├── .env                          # API keys (not committed)
└── chatbot.db                    # SQLite chat history (auto-created)
Design Decisions

Why RAG over fine-tuning? RAG grounds answers in retrieved source text, is cheap to update (just add documents), and dramatically reduces hallucination for factual Q&A — no retraining required.

Why chunk with overlap? A 200-character overlap ensures ideas split across chunk boundaries still appear intact in at least one chunk, preventing loss of context mid-sentence.

Why FAISS? Fast exact similarity search that runs entirely in-process — no external vector database service to host or pay for.

Why LangGraph over a simple chain? The conditional edge (tools_condition) lets the LLM decide at runtime whether a tool is needed and loop back after tool execution — enabling multi-step reasoning that a linear chain can't express.

Why per-thread retrievers? Each conversation maintains its own document index, so uploading a PDF in one chat doesn't leak context into another.

Known Limitations
Local models (llama3.2) are less reliable at tool selection than frontier models; qwen2.5 performs better if routing misfires
SQLite persistence is single-machine; a hosted DB would be needed for multi-user deployment
FAISS indexes live in memory and rebuild on restart
Future Improvements
 Persist FAISS indexes to disk so documents survive restarts
 Add source citations with page numbers in the UI
 Support multiple document formats (DOCX, TXT, web URLs)
 Add retrieval evaluation metrics (hit rate, MRR)
 Deploy with a hosted API model
