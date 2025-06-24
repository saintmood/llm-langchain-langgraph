import os

from dotenv import load_dotenv
from langchain_community.document_loaders import (
    TextLoader,
)  # see the official documentation for more loaders
from langchain_text_splitters import (
    CharacterTextSplitter,
)  # this is needed for splitting text into chunks, since LLM has token limits
from langchain_openai import (
    OpenAIEmbeddings,
)  # this entity transforms text into embeddings - a group of vectors that represent the text
from langchain_pinecone import PineconeVectorStore

load_dotenv()


if __name__ == "__main__":
    print("Ingesting")
    loader = TextLoader("medium_blog.txt", encoding="utf8")  # load the text file
    document = loader.load()

    print("Splitting text into chunks")
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=0
    )  # split the text into chunks of 1000 characters with an overlap of 0 characters
    # chunks should be smaller enough to fit into the LLM context window
    texts = text_splitter.split_documents(document)  # split the document into chunks
    print(f"Number of chunks: {len(texts)}")

    embeddings = OpenAIEmbeddings(
        api_key=os.environ["OPENAI_API_KEY"]
    )  # create embeddings for the text chunks
    print("Ingesting ...")
    PineconeVectorStore.from_documents(
        texts,
        embeddings,
        index_name=os.environ["INDEX_NAME"],
    )  # create a vector store from the text chunks and embeddings
    print("Finish the ingestion process")
