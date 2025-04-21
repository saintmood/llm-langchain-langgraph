import requests
from langchain.llms import BaseLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Define the base URL and model name
BASE_URL = "http://localhost:1234"  # Replace with your API URL
MODEL_NAME = "meta-llama-3.1-8b-instruct"  # Replace with your model name


class MetaLlama3LLM(BaseLLM):
    def __init__(self, base_url, model_name):
        self.base_url = base_url
        self.model_name = model_name

    def _call(self, prompt, stop=None):
        # Send a POST request to the Meta LLaMA 3 API
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={"model": self.model_name, "prompt": prompt},
        )
        if response.status_code == 200:
            return response.json().get("response", "")
        else:
            raise ValueError(f"Error: {response.status_code}, {response.text}")

    def _generate(self, prompt, stop=None):
        """Implements the abstract method _generate."""
        output = self._call(prompt, stop)
        return {"text": output}

    @property
    def _llm_type(self):
        """Defines the type of the LLM."""
        return MODEL_NAME


if __name__ == "__main__":
    print("Hello, LangChain!")
    # Initialize the Meta LLaMA 3 LLM
    meta_llama_llm = MetaLlama3LLM(BASE_URL, MODEL_NAME)

    # Create a prompt template
    prompt_template = PromptTemplate(
        input_variables=["question"],
        template="Question: {question}\nAnswer: Let's think step by step.",
    )

    # Build the LangChain
    chain = LLMChain(llm=meta_llama_llm, prompt=prompt_template)

    # Run the chain with a sample question
    response = chain.run({"question": "What are the key features of Meta LLaMA 3?"})
    print(response)
