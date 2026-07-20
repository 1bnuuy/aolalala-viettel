from app.rag.retriever import retrieve

while True:
    question = input("You: ")

    if question.lower() in ("exit", "quit"):
        break

    docs = retrieve(question)

    print("\nRetrieved Context:\n")

    for i, doc in enumerate(docs, start=1):
        print(f"Chunk {i}:")
        print(doc.page_content)
        print("-" * 40)