from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.openai import ChatOpenAI

from third_parties.linkedin import scrape_linkedin_profile

if __name__ == "__main__":
    load_dotenv()
    print("Hello LangChain!")

    summary_template = """
    given the linkedin information {information} about a person I want you to create:
    1. A short summary of the person
    2. Two interesting facts about the person
    """
    # Define the prompt template
    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
    )
    # Initialize the language model
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    # Create the LLM chain
    chain = summary_prompt_template | llm
    linkedin_data = scrape_linkedin_profile(
        linkedin_profile_url="https://www.linkedin.com/in/misha-shchetinin-1042a942/"
    )
    # Run the chain with a sample name
    res = chain.invoke({"information": linkedin_data})
    print(res)
