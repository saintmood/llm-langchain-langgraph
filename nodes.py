from dotenv import load_dotenv
from langgraph.graph import MessagesState  # hold both human messages and AI messages
from langgraph.prebuilt import (
    ToolNode,
)  # parse the messages, and if the last message is a tool call, it will execute the tool and return the result

from react import llm, tools

load_dotenv()


SYSTEM_MESSAGE = (
    """You are a helpful assistant that can use tools to answer questions."""
)


def run_agent_reasoning(state: MessagesState) -> MessagesState:
    """
    Run the agent reasoning step.
    This function will parse the messages, and if the last message is a tool call,
    it will execute the tool and return the result.
    """
    response = llm.invoke(
        [{"role": "system", "content": SYSTEM_MESSAGE}], *state["messages"]
    )
    return {
        "messsages": [response]
    }  # langgraph is going to append the response to it's state.


tool_node = ToolNode(tools)
