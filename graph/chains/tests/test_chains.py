from pprint import pprint

from dotenv import load_dotenv

load_dotenv()


from graph.chains.generation import generation_chain
from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from graph.chains.hallucination_grader import hallucination_grader, GradeHallucination
from ingestion import retriever


def test_retrieval_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": question, "document": doc_txt}
    )

    assert res.binary_score == "yes"


def test_retrieval_grader_answer_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": "how to raise a monster?", "document": doc_txt}
    )

    assert res.binary_score == "no"


def test_generation_chain() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"context": docs, "question": question})
    pprint(generation)


def test_hallucination_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content

    generation = generation_chain.invoke({"context": docs, "question": question})

    res: GradeHallucination = hallucination_grader.invoke(
        {"documents": doc_txt, "generation": generation}
    )

    assert res.binary_score is True


def test_hallucination_grader_answer_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)

    res: GradeHallucination = hallucination_grader.invoke(
        {"documents": docs, "generation": "Just a random string."}
    )

    assert res.binary_score is False
