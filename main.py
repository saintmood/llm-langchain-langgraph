from typing import Any

from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.agents.tools import Tool
from langchain_experimental.agents import create_csv_agent
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain_openai import ChatOpenAI

load_dotenv()


def main():
    print("Start ...")
    instructions = """You are an agent designed to write and execute python code to answer questions.
    You have access to a python REPL, which you can use to execute code.
    If you get an error, debug your code and try again.
    Only use the output of your code to answer the question.
    You might know the answer without running any code, but you should always try to run code first.
    If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
    """

    base_prompt = hub.pull("langchain-ai/react-agent-template")
    prompt = base_prompt.partial(instructions=instructions)
    tools = [PythonREPLTool()]
    agent = create_react_agent(
        prompt=prompt,
        llm=ChatOpenAI(temperature=0, model="gpt-4-turbo"),
        tools=tools,
    )
    python_agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    python_agent_executor.invoke(
        input={
            "input": """generate and save in current working directory 15 QRCodes 
                                 that point to www.udemy.com/cources/langchain, you have qrcode package installed already"""
        }
    )

    csv_agent_executor = create_csv_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-4"),
        path="episode_info.csv",
        verbose=True,
    )
    csv_agent_executor.invoke(
        input={"input": "How many columnts are there in the file episode_info.csv?"}
    )

    def python_agent_executor_wrapper(original_prompt: str) -> dict[str, Any]:
        """Wrapper for the Python REPL tool to execute code."""
        return python_agent_executor.invoke(
            input={"input": original_prompt}
        )
    
    # Router Agent Section
    tools = [
        Tool(
            name="Python Agent",
            func=python_agent_executor_wrapper,
            description="""Useful when you need to tranform natural language to python and execute the python code,
            returning the results of the code execution
            DOES NOT ACCEPT CODE AS INPUT""",
        ),
        Tool(
            name="CSV Agent",
            func=csv_agent_executor.invoke,
            description="""Useful when you need to answer questions about the content of a CSV file.
            The input should be a question about the CSV file, not the file itself.""",
        ),
    ]

    prompt = base_prompt.partial(instructions="")
    grand_agent = create_react_agent(
        prompt=prompt,
        llm=ChatOpenAI(temperature=0, model="gpt-4-turbo"),
        tools=tools,
    )
    grand_agent_executor = AgentExecutor(agent=grand_agent, tools=tools, verbose=True)
    print(
        grand_agent_executor.invoke(
            input={
                "input": """How many columnts are there in the file episode_info.csv?"""
            }
        )j
    )


if __name__ == "__main__":
    main()
