# llm-langchain-langgraph


## Section 12: LangGraph
This time we will build a project - ReActAgentExecutor, but with langgraph.
Same impleemntation has been done in scope of Section 4.
This time we are implemention the tool calling via ChatOpenAI.bind_tools method. This is a vendor's implementation,
when the tool's description will be eximined and correct tool will be returned in the special key in the response.
This is an oposite approach to ReAct.