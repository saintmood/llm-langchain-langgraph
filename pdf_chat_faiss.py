import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub

load_dotenv()


if __name__ == "__main__":
    print("Hi")
    pdf_path = "2210.03629v3.pdf"
    loader = PyPDFLoader(pdf_path)  # load the PDF file
    documents = (
        loader.load()
    )  # load the document. At the moment there is no control over the chunk size
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=30, separator="\n"
    )
    docs = text_splitter.split_documents(documents=documents)
    embeddings = OpenAIEmbeddings()
    vectorestore = FAISS.from_documents(docs, embeddings)
    vectorestore.save_local(
        "faiss_index_react"
    )  # save the vector store to a local directory
    new_vectorestore = FAISS.load_local(
        "faiss_index_react", embeddings, allow_dangerous_deserialization=True
    )
    retrieval_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combined_docs_chain = create_stuff_documents_chain(OpenAI(), retrieval_chat_prompt)
    retrieval_chain = create_retrieval_chain(
        new_vectorestore.as_retriever(), combined_docs_chain
    )
    res = retrieval_chain.invoke({"input": "Give me the gist of ReAct in 3 sentences"})
    print(res["answer"])
