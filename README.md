# llm-langchain-langgraph


## Advanced RAG implementation.
For the reference Eden's project on github: https://github.com/emarco177/langgraph-course/tree/project/agentic-rag
More details can be found from the inspirational materials: https://github.com/mistralai/cookbook/tree/main/third_party/langchain

The implementaiton is base on 3 papers:
- Self-RAG: Learning to Retrive, Generate, and Critique throught self-reflection
- Corrective Retrieval Augmented Generation ( C-RAG )
- Adaptive-Rag: Learning to adapt retrieval-augmented LLM through Question Complexity


General idea of these papers is to add a reflection into the flow.


### Self-RAG 
The second part of the Section 15 is Self-Rag implementation.


### Adaptive Rag
The essence of the implementation is to use a question router,which points to a different parts of the flow.
The router will check if the user question can be answered from the documents we have in the vector store. And if it's not possible, then it will route to a web search.