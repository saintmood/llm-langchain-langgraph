from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM

information = """
"""

if __name__ == "__main__":
    print("Hello, Langchain!")

    summary_template = """
        given the information {information} about a person from I want you to create:
        1. a short summary
        2. two interesting facts about them
    """

    sumary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )
    # llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    llm = OllamaLLM(
        base_url="http://127.0.0.1:1234",
        model="meta-llama-3.1-8b-instruct",
        temperature=0.7,
    )
    chain = sumary_prompt_template | llm
    res = chain.invoke(input={"information": information})
    print(res)
