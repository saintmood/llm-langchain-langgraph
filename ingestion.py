import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore


embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)
load_dotenv()


def ingest_docs():
    document_path = "langchain-docs/api.python.langchain.com/en/latest"
    loader = ReadTheDocsLoader(
        document_path,
        encoding="utf-8",
    )
    raw_docs = loader.load()
    print("Loaded documents count:", len(raw_docs))
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=15)
    # it's common to pick a chunk size that the length of the context equals to 4 or 5 but not meore.
    documents = text_splitter.split_documents(raw_docs)
    print("Split documents count:", len(documents))
    for doc in documents:
        new_url = doc.metadata.get("source")
        new_url = new_url.replace("\\", "/").replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})
    PineconeVectorStore.from_documents(
        documents,
        embeddings,
        index_name=os.getenv("INDEX_NAME"),
        embeddings_chunk_size=300,
    )


if __name__ == "__main__":
    ingest_docs()
