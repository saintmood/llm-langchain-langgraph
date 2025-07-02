from backend.core import run_llm
import streamlit as st


st.header("LangChain Udemy Course - Documentation Helper Bot")
prompt = st.text_input("Prompt", "What is LangChain?")

if (
    "chat_answers_history" not in st.session_state
    and "user_prompt_history" not in st.session_state
    and "chat_history" not in st.session_state
):
    st.session_state.chat_answers_history = []
    st.session_state.user_prompt_history = []
    st.session_state.chat_history = []


def create_sources_string(sources):
    """
    Create a formatted string of sources from the set of source URLs.
    """
    if not sources:
        return "No sources found."
    sources_string = "Sources:\n"
    for source in sources:
        sources_string += f"- {source}\n"
    return sources_string


if prompt:
    with st.spinner("Generating response..."):
        generated_response = run_llm(prompt)
        sources = set(
            [doc.metadata["source"] for doc in generated_response["source_documents"]]
        )
        formatted_response = (
            f"{generated_response['result']} \n\n {create_sources_string(sources)}"
        )
        st.session_state.user_prompt_history.append(prompt)
        st.session_state.chat_answers_history.append(formatted_response)

if st.session_state.chat_answers_history:
    for generated_response, user_query in zip(
        st.session_state.chat_answers_history, st.session_state.user_prompt_history
    ):
        st.chat_message("User").write(user_query)
        st.chat_message("Assistant").write(generated_response)
