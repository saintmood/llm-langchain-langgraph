import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.runnables import RunnablePassthrough

load_dotenv()


def format_docs(docs):
    """Format the documents for the RAG prompt."""
    return (
        "\n\n".join([doc.page_content for doc in docs])
        if docs
        else "No relevant documents found."
    )


if __name__ == "__main__":
    print("Retrieving...")
    embeddings = (
        OpenAIEmbeddings()
    )  # the embeddings model used to convert text into vectors
    llm = ChatOpenAI()

    query = "What is Pinecone in machine learning?"
    chain = PromptTemplate.from_template(template=query) | llm
    result = chain.invoke({})
    print(result.content)

    vectorstore = PineconeVectorStore(
        index_name=os.environ["INDEX_NAME"],
        embedding=embeddings,
    )
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_documents_chain = create_stuff_documents_chain(
        llm, retrieval_qa_chat_prompt
    )
    retrieval_chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(),
        combine_docs_chain=combine_documents_chain,
    )
    # by now, the execution flow would be:
    # 1. Take the user prompt and transform it into a vector using the embeddings model
    # 2. Send this embedded prompt to the vector store to retrieve relevant documents as a context
    # 3. Plug the context into the rag prompt
    # 4. send the prompt to the LLM to get the final answer
    result = retrieval_chain.invoke(input={"input": query})
    print(result)

    template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use three sentences maximum and keed the answer concise as possible.
    Always say "thanks for asking!" at the end of your answer.
    
    {context}
    
    Question: {question}
    
    Helpful Answer:"""
    custome_rag_prompt = PromptTemplate(template=template)
    rag_chain = (
        {
            "context": vectorstore.as_retriever() | format_docs,
            "question": RunnablePassthrough(),  # on it's own the RunnablePassthrough will just pass the input unchanged to the next key in the map
        }
        | custome_rag_prompt
        | llm
    )
    res = rag_chain.invoke(query)
    print(res)
