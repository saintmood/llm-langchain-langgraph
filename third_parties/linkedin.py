import os

import requests
from dotenv import load_dotenv


load_dotenv()


def scrape_linkedin_profile(linkedin_profile_url: str, mock: bool = False) -> str:
    """
    scrape information from a LinkedIn profiles,
    Manually scrape the information from the LinkedIn profile page."""

    if mock:
        linkedin_profile_url = ""
        response = requests.get(linkedin_profile_url, timeout=10)
    else:
        api_endpoint = ""
        params = {
            "linkedInUrl": linkedin_profile_url,
            "apikey": os.getenv("SCRAPIN_API_KEY"),
        }
        response = requests.get(api_endpoint, params=params, timeout=10)
    data = response.json().get("person")
    data = {
        k: v
        for k, v in data.items()
        if v not in ([], "", None) and k not in ["certifications"]
    }
    return data


if __name__ == "__main__":
    print(
        scrape_linkedin_profile(
            linkedin_profile_url="https://www.linkedin.com/in/misha-shchetinin-1042a942/"
        )
    )
