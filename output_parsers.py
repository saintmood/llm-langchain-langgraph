from typing import Any, Dict, List, Union

from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


class Summary(BaseModel):
    summary: str = Field(
        description="A short summary of the person based on their LinkedIn profile."
    )
    facts: List[str] = Field(description="Two interesting facts about the person.")

    def to_dict(self) -> Dict[str, Union[str, List[str]]]:
        """Convert the Summary object to a dictionary."""
        return {"summary": self.summary, "facts": self.facts}


summary_parser = PydanticOutputParser(pydantic_object=Summary)
