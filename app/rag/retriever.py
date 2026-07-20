from app.rag.embeddings import get_embeddings
from app.rag.vectorstore import load_vectorstore

# k = 1 gives 1 nearest chunk (answer) to the question
def retrieve(question: str, k: int = 1): 
    embeddings = get_embeddings()
    db = load_vectorstore(embeddings)

    return db.similarity_search(question, k=k)