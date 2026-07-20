from langchain_community.document_loaders import TextLoader


def load_documents():
    loader = TextLoader("documents/faq.md")
    return loader.load()