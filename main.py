from dotenv import load_dotenv
from langchain.agents import tool, Tool
from langchain.prompts import PromptTemplate
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.tools.render import render_text_description
from langchain_openai import ChatOpenAI
from langchain.agents.format_scratchpad.log import format_log_to_str
from langchain.agents.output_parsers.react_single_input import (
    ReActSingleInputOutputParser,
)

from callbacks import AgentCallbackHandler

load_dotenv()


@tool
def get_text_length(text: str) -> int:
    # when writing tool it's importatnt to add a docstring
    # with a description of tool's perpose.
    # so the LLM can understand what it does and pick up the correct tool by evaluating the docstring.
    """Returns the length of the given text by charactes."""
    text = text.strip("'\n").strip('"')
    return len(text)


def find_tool_by_name(tools, tool_name: str) -> Tool:
    """Finds a tool by its name."""
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool with name {tool_name} not found.")


if __name__ == "__main__":
    print("Hello, World!")
    tools = [get_text_length]

    # on the schematic diagram this prompt will be sent to the LLM.
    template = """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought: {agent_scratchpad}
    """
    prompt = PromptTemplate(template=template).partial(
        tools=render_text_description(tools),
        tool_names=", ".join([tool.name for tool in tools]),
    )
    intermediate_steps = []
    llm = ChatOpenAI(
        temperature=0.0,
        model="gpt-4o-mini",
        model_kwargs={"stop": ["\nObservation", "Observation", "Observation:"]},
        callbacks=[AgentCallbackHandler()],
    )

    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"]),
        }
        | prompt
        | llm
        | ReActSingleInputOutputParser()
    )
    agent_step = ""
    while not isinstance(agent_step, AgentFinish):
        agent_step: AgentAction = agent.invoke(
            {
                "input": "What is the length of the text 'Dog' in characters?",
                "agent_scratchpad": intermediate_steps,
            }
        )
        print(agent_step)

        if isinstance(agent_step, AgentAction):
            tool_name = agent_step.tool
            tool_to_use = find_tool_by_name(tools, tool_name)
            tool_input = agent_step.tool_input
            observation = tool_to_use.func(str(tool_input))
            print(f"Observation: {observation}")
            intermediate_steps.append(
                (
                    agent_step,
                    observation,
                )
            )
    if isinstance(agent_step, AgentFinish):
        print("Final Answer:", agent_step.return_values)
