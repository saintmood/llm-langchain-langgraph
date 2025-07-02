import os

from dotenv import load_dotenv
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains.history_aware_retriever import create_history_aware_retriever

load_dotenv()


def run_llm(query: str, chat_history: list):
    """
    Run the LLM with the given query.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)
    docsearch = PineconeVectorStore(
        index_name=os.getenv("INDEX_NAME"), embedding=embeddings
    )
    chat = ChatOpenAI(verbose=True, model="gpt-4o", temperature=0.0)

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)

    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
    history_aware_retriever = create_history_aware_retriever(
        llm=chat, retriever=docsearch.as_retriever(), prompt=rephrase_prompt
    )

    qa = create_retrieval_chain(
        history_aware_retriever, combine_docs_chain=stuff_documents_chain
    )
    result = qa.invoke(input={"input": query, "chat_history": chat_history})
    response = {
        "query": result["input"],
        "result": result["answer"],
        "source_documents": result["context"],
    }
    return response


if __name__ == "__main__":
    query = "What is LangChain?"
    print(run_llm(query)["result"])
