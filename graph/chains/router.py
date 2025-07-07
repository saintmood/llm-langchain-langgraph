# this is implementaion of router chain
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class RouteQuery(BaseModel):
    """Route query to the most relevant datasource."""

    datasource: Literal["vectorestore", "websearch"] = Field(
        ...,
        description="Given a user question choose to route it to vectorestore or web search.",
    )


llm = ChatOpenAI(temperature=0)
structured_llm_router = llm.with_structured_output(RouteQuery)


system = """You are an expert at routing a user question to a vectorstore or web search. \n
The vectorstore contains documents related to agents, prompt engineering and adversarial attacks. \n
Use the vcetorstore for questions on these topics. For all else, use web search."""

router_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question}"),
    ]
)

question_router = router_prompt | structured_llm_router
