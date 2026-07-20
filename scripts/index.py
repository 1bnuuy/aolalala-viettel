from app.rag.loader import load_documents
from app.rag.splitter import split_documents
from app.rag.embeddings import get_embeddings
from app.rag.vectorstore import create_vectorstore

documents = load_documents()
chunks = split_documents(documents)

print(f"Chunks: {len(chunks)}")

embeddings = get_embeddings()

create_vectorstore(chunks, embeddings)