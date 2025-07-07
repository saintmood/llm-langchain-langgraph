from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from graph.chains.answer_grader import answer_grader
from graph.chains.hallucination_grader import hallucination_grader
from graph.consts import GENERATE, GRADE_DOCUMENTS, RETRIEVE, WEBSEARCH
from graph.nodes import generate, grade_documents, retrieve, web_search
from graph.state import GraphState

load_dotenv()


def decide_to_generate(state):
    if state["web_search"]:
        return WEBSEARCH
    return GENERATE


def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    score = hallination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    if hallucination_grade := score.binary_score:
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE QUESTION vs GENERATION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        if answer_grade := score.binary_score:
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"  # later this will be used in the langgraph mapping.
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS---")
        return "not supported"


workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(WEB_SEARCH, web_search)
workflow.add_node(GENERATE, generate)

workflow.set_entry_point(RETRIEVE)
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workfflow.add_conditional_edges(
    GRADE_DOCUMENTS, decide_to_generate, {WEBSEARCH: WEBSEARCH, GENERATE: GENERATE}
)
workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    {"useful": END, "not useful": WEB_SEARCH, "not supported": GENERATE},
)
workflow.add_edge(WEBSEARCH, GENERATE)
workflow.add_edge(GENERATE, END)
app = workflow.compile()
