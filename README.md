##Multi-Utility RAG Chatbot

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

-----------------------------------------------------------------------------------------------------------------------------------------------------------------
#Architecture

<img width="879" height="385" alt="image" src="https://github.com/user-attachments/assets/9cac48cd-8898-4271-acfa-a8180814ed79" />


-------------------------------------------------------------------------------------------------------------------------------------------------------------------

#RAG pipeline: 
PDF → PyPDFLoader → RecursiveCharacterTextSplitter (1000 chars, 200 overlap) → embeddings → FAISS vector store → similarity retrieval (top-k=4) → context injected into the LLM prompt.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Tech Stack

<img width="743" height="412" alt="image" src="https://github.com/user-attachments/assets/4729c66e-80bc-4dd6-ac60-b5c8bac2856d" />


-------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Setup
1. Clone and install
bash

git clone https://github.com/tanishbaisla/Multi-Utility-Chatbot.git

cd Multi-Utility-Chatbot

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

-------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Usage

Ask anything: Type a question and get an answer from the model's knowledge
Attach a PDF: Click the attach icon in the chat bar, upload, then ask about it
Switch conversations: Click any title in the sidebar
Delete a conversation: Click the 🗑 next to it
Using an API model instead of Ollama

To use Google Gemini (or OpenAI) instead of local models, replace the LLM setup in langraph_rag_backend.py:

python
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

Add your key to a .env file:

GOOGLE_API_KEY=your_key_here

Note: switching embedding providers changes vector dimensions. Delete chatbot.db and re-upload your PDFs after switching.
-------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Design Decisions

Why RAG over fine-tuning? 

RAG grounds answers in retrieved source text, is cheap to update (just add documents), and dramatically reduces hallucination for factual Q&A no retraining required.

Why chunk with overlap? 

A 200-character overlap ensures ideas split across chunk boundaries still appear intact in at least one chunk, preventing loss of context mid-sentence.

Why FAISS? 

Fast exact similarity search that runs entirely in-process no external vector database service to host or pay for.

Why LangGraph over a simple chain? 

The conditional edge (tools_condition) lets the LLM decide at runtime whether a tool is needed and loop back after tool execution enabling multi-step reasoning that a linear chain can't express.

Why per-thread retrievers? 

Each conversation maintains its own document index, so uploading a PDF in one chat doesn't leak context into another.
