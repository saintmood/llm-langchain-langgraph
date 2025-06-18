from typing import Tuple

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent
from output_parsers import Summary, summary_parser
from third_parties.linkedin import scrape_linkedin_profile


def ice_break_with(name: str) -> Tuple[Summary, str]:
    linkedin_url = linkedin_lookup_agent(name=name)
    linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_url)

    summary_template = """
    given the linkedin information {information} about a person I want you to create:
    1. A short summary of the person
    2. Two interesting facts about the person

    Use information from LinkedIn
    \n{format_instructions}
    """
    # Define the prompt template
    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        partial_variables={
            "format_instuctions": summary_parser.get_format_instructions()
        },
        template=summary_template,
    )
    # Initialize the language model
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    # chain = summary_prompt_template | llm
    chain = summary_prompt_template | llm | summary_parser
    linkedin_data = scrape_linkedin_profile(
        linkedin_profile_url="https://www.linkedin.com/in/misha-shchetinin-1042a942/"
    )
    # Run the chain with a sample name
    res: Summary = chain.invoke({"information": linkedin_data})
    return res, linkedin_data.get("photoUrl")


if __name__ == "__main__":
    load_dotenv()
    print("Ice Breaker Enter")
    ice_break_with(name="Misha Shchetinin Capgemini")
