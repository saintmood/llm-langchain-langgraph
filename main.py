import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, MessagesState, END

from nodes import run_agent_reasoning, tool_node

load_dotenv()


AGENT_REASON = "agent_reason"
ACT = "act"
LAST = -1


def should_continue(state: MessagesState) -> str:
    if state.messages[LAST].tool_calls:
        # If the last message is a tool call, we continue to the ACT node.
        return ACT
    return END


flow = StateGraph(MessagesState)
flow.add_node(AGENT_REASON, run_agent_reasoning)
flow.set_entry_point(AGENT_REASON)
flow.add_node(ACT, tool_node)

flow.add_conditional_edges(
    AGENT_REASON, should_continue, {END: END, ACT: ACT}
)  # the mapping defines explicitly which node to go after the reasoning node.

flow.add_edge(ACT, AGENT_REASON)  # after the tool call, we go back to reasoning.

app = flow.compile()
app.get_graph().draw_mermaid_png(output_file_path="flow.png")

if __name__ == "__main__":
    print("Hello ReAct LangGraph with Function Calling!")
    res = app.invoke(
        {
            "messages": [
                HumanMessage(
                    content="What is the weather in Tokyo? List it and then triple it."
                )
            ]
        }
    )
    print(res["messages"][LAST].content)
