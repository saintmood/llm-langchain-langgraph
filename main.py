from typing import List, Sequence

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, MessageGraph

from chains import generation_prompt, reflection_prompt

load_dotenv()


REFLECT = "reflect"
GENERATE = "generate"


def generation_node(state: Sequence[BaseMessage]):
    return generation_prompt.invoke({"messages": state})


def reflection_node(state: Sequence[BaseMessage]):
    """Masking the AI as a human to start the conversation"""
    res = reflection_prompt.invoke({"messages": state})
    return [HumanMessage(content=res.content)]


def should_continue(state: Sequence[BaseMessage]) -> str:
    """Here in the real case the LLM should perform the evaluation of the state and decide whether to continue or not."""
    if len(state) > 6:
        return END
    return REFLECT


builder = MessageGraph()
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)  # set the generation node as the __start__ node.
builder.add_conditional_edges(GENERATE, should_continue)
builder.add_edge(REFLECT, GENERATE)  # after reflection, we go back to generation


graph = builder.compile()


if __name__ == "__main__":
    print("Hello, World!")
    inputs = HumanMessage(content="Make this tweet better: paste the tweet here")

    response = graph.invoke(inputs)
