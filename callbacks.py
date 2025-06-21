from langchain.callbacks.base import BaseCallbackHandler


class AgentCallbackHandler(BaseCallbackHandler):
    """
    Custom callback handler for the agent.
    This class can be extended to implement custom behavior during the agent's execution.
    """

    def on_llm_start(
        self,
        serialized,
        prompts,
        *,
        run_id,
        parent_run_id=None,
        tags=None,
        metadata=None,
        **kwargs,
    ):
        print(f"***Prompt to LLM was:***\n{prompts[0]}")
        print(9 * "*")

    def on_llm_end(self, response, *, run_id, parent_run_id=None, **kwargs):
        print(f"***LLM response:***\n{response.generations[0][0].text}")
        print(9 * "*")
