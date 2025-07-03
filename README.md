# llm-langchain-langgraph


## Section 7: Own Implementation of GPT Code-Interpreter.
The project is able to write and to run code. 
Features to be reviewed in scope of section-7:
- PythonREPL Agent
- CSV Agent
- Router Agent
- OpenAI Functions

### Code Interpreter
This time the code interpreter would be another tool. This tool generates the python code and then execute this code.

The implementation of chat should be able to parse income data as a csv, perform sorting and filtering operations. 

This requires additional security protection, since this is remote code execution literraly. If the intruder deliver the exploit and 
modify the prompt the exploited code, which will be executed by the agnet will cause the security breach.