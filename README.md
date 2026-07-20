# Hospital Chatbot (RAG)

A Retrieval-Augmented Generation (RAG) chatbot built with **Python**, **LangChain**, **FAISS**, and **FastAPI**. The chatbot retrieves relevant information from hospital documents and uses an LLM to generate accurate responses.

---

## Features

* 📄 Load Markdown, PDF, and other supported documents
* ✂️ Automatic document chunking
* 🧠 SentenceTransformer embeddings
* 🔍 FAISS vector search
* 🤖 LLM-powered responses
* ⚡ FastAPI backend
* 🧪 Terminal chat for local testing

---

## Tech Stack

* Python 3.12+
* FastAPI
* LangChain
* FAISS
* Sentence Transformers
* Hugging Face
* Uvicorn

---

## Project Structure

```text
chatbot-ai/
│
├── app/
│   ├── api/
│   │   └── chat.py
│   │
│   ├── rag/
│   │   ├── loader.py
│   │   ├── splitter.py
│   │   ├── embeddings.py
│   │   ├── vectorstore.py
│   │   └── retriever.py
│   │
│   ├── services/
│   │   └── chat_service.py
│   │
│   ├── config.py
│   └── main.py
│
├── documents/
│
├── scripts/
│   ├── index.py
│   └── chat.py
│
├── vectorstore/
│
├── requirements.txt
├── .env
└── README.md
```

---

## Installation

Clone the repository.

```bash
git clone <repository-url>
cd chatbot-ai
```

Create a virtual environment.

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies.

```bash
pip install -r requirements.txt
```

---

## Add Documents

Place your knowledge base inside the `documents` folder.

Example:

```text
documents/
├── faq.md
├── hospital.pdf
└── doctors.md
```

---

## Build the Vector Database

Whenever documents are added or modified, rebuild the embeddings.

```bash
python -m scripts.index
```

A FAISS vector database will be generated inside:

```text
vectorstore/
```

---

## Test Retrieval

Run the terminal chat.

```bash
python -m scripts.chat
```

Example:

```text
You: What are the visiting hours?

Retrieved Context:

Visiting hours:
Monday - Friday: 8:00 AM - 5:00 PM
```

---

## Run the API

Start the FastAPI server.

```bash
uvicorn app.main:app --reload
```

Default address:

```text
http://127.0.0.1:8000
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

---

## RAG Pipeline

```text
Documents
    │
    ▼
Document Loader
    │
    ▼
Text Splitter
    │
    ▼
Embeddings
    │
    ▼
FAISS Vector Store
    │
    ▼
Retriever
    │
    ▼
LLM
    │
    ▼
Response
```

---

## Development Workflow

1. Add or update documents.
2. Run `python -m scripts.index`.
3. Test retrieval with `python -m scripts.chat`.
4. Start the FastAPI server.
5. Connect the frontend or backend.

---

## Future Improvements

* Streaming responses
* Conversation memory
* Multiple document loaders
* Metadata filtering
* Hybrid search (BM25 + Vector Search)
* Reranking
* Docker support
* Authentication
* Persistent chat history
* Evaluation and benchmarking

---

## License

MIT License
