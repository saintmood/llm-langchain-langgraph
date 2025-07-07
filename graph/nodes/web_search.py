from typing import Any, Dict

from dotenv import load_dotenv
from langchain.schema import Document
from langchain_tavily import TavilySearch

from graph.state import GraphState

load_dotenv()
web_search_tool = TavilySearch(max_results=3)


def web_search(state: GraphState) -> Dict[str, Any]:
    print("---Web Search---")
    question = state["question"]
    documents = state["documents"]

    tavily_results = web_search_tool.invoke(question)

    # combine 3 results into 1 Document
    joined_tavily_result = "\n".join([tr["content"] for tr in tavily_results])
    web_result = Document(page_content=joined_tavily_result)
    if documents is not None:
        documents.append(web_result)
    else:
        documents = [web_result]
    return {"documents": documents, "question": question}


if __name__ == "__main__":
    web_search(state={"question": "agent memory", "documents": None})
