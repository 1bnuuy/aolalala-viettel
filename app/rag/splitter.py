from langchain_core.documents import Document

def split_documents(documents):
    chunks = []

    for doc in documents:
        sections = doc.page_content.split("-------------------------")

        for section in sections:
            section = section.strip()
            if section:
                chunks.append(Document(page_content=section))

    return chunks