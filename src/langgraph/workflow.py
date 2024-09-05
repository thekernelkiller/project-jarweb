from typing import Dict, Any, List, Optional
from langchain_groq import ChatGroq
from langchain.embeddings import OllamaEmbeddings
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.workflow import StateGraph, START, END, add_messages
from crewai_tools import CodeInterpreterTool, SeleniumScrapingTool, BaseTool
from langchain.vectorstores import Chroma
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.chains import create_retrieval_chain, LLMChain
from tools.rag_tool import RAGTool
import os


embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectordb = Chroma(embedding_function=embeddings, persist_directory="./data")
retriever = vectordb.as_retriever()
qa = create_retrieval_chain(ChatGroq(model="llama-3.1-8b-instant", temperature=0), retriever, chain_type="stuff")

selenium_code_interpreter = CodeInterpreterTool()
selenium_scraping_tool = SeleniumScrapingTool()

class LangGraphState(Dict):
    """State for the LangGraph web automation workflow."""
    webpage_data: str = ""
    generated_xpath: str = ""
    generated_code: str = ""
    execution_result: str = ""
    task_complete: bool = False
    messages: Optional[List[BaseMessage]] = None
    user_prompt: str = ""

def extract_url_from_prompt(user_prompt: str) -> str:
    """Extract the target webpage URL from the user prompt using an LLM."""
    prompt_template = PromptTemplate(
        input_variables=["user_prompt"],
        template="""You are a web automation expert. Given the user prompt: '{user_prompt}' extract the URL of the website to be automated. 
                   If no URL is found, return an empty string."""
    )
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    chain = LLMChain(llm=llm, prompt=prompt_template)
    url = chain.run(user_prompt=user_prompt)
    return url.strip()

def generate_xpath_expression(state: LangGraphState) -> str:
    """Generate an XPath expression for the target element."""
    webpage_data = state["webpage_data"]
    user_prompt = state['user_prompt']

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are an expert in web automation and XPath generation. "
                  "You can create XPaths to locate elements on webpages."),
        ("user", f"Here's the relevant HTML content from the webpage:\n\n{webpage_data}\n\n"
                f"The user wants to perform these actions:\n\n{user_prompt}\n\n"
                "Generate a single, accurate XPath expression that identifies the specific "
                "element the user wants to interact with based on their instructions.")
    ])

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    xpath_chain = prompt_template | llm | StrOutputParser() 
    xpath = xpath_chain.invoke({"input": ""}) 

    return xpath 

def generate_code_from_xpath(xpath: str, user_prompt: str) -> str:
    """
    Generate Selenium Python code based on the XPath and user prompt.

    This function constructs Selenium code to interact with the element 
    identified by the XPath, interpreting actions from the user prompt.
    """
    code_template = """
from selenium import webdriver
from selenium.webdriver.common.by import By
import time

driver = webdriver.Chrome()
driver.get('{url}')

element = driver.find_element(By.XPATH, '{xpath}')
{actions}

time.sleep(2)
driver.quit()
    """

    url = extract_url_from_prompt(user_prompt)
    actions = _generate_selenium_actions(user_prompt, xpath)
    
    return code_template.format(url=url, xpath=xpath, actions=actions)

def _generate_selenium_actions(user_prompt: str, xpath: str) -> str:
    """
    Generates Selenium actions based on the user prompt. 
    """
    prompt_template = PromptTemplate(
        input_variables=["user_prompt", "xpath"],
        template="""You are a Selenium expert. Given the user prompt: '{user_prompt}' and the XPath: '{xpath}' generate ONLY the python code to interact with the element, you don't need to include the `from selenium...` and the driver definitions. Just the actions on the element.
                   Make sure to handle edge cases and that the code is executable. Don't include any comments."""
    )
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    chain = LLMChain(llm=llm, prompt=prompt_template)
    actions = chain.run(user_prompt=user_prompt, xpath=xpath)
    return actions

class TaskCompletionOutput(Dict):
    """Output structure for TaskCompletionTool."""
    task_complete: bool
    reason: str

class TaskCompletionTool(BaseTool):
    """
    A tool that uses an LLM to determine if a web automation task is complete.
    """
    name = "Task Completion Checker"
    description = (
        "Useful for determining if a web automation task has been successfully completed. "
        "Input should be the original user prompt, the output from the Selenium "
        "code execution, and the current content of the webpage."
    )

    def _run(self, user_prompt: str, execution_result: str, webpage_content: str) -> TaskCompletionOutput:
        """
        Uses an LLM to analyze the prompt, execution result, and webpage 
        content to determine task completion.
        """
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are an expert in evaluating web automation task completion. "
                      "Analyze the user's request, the result of the code execution, "
                      "and the webpage content to determine if the task was successful."),
            ("user", f"Here's the original user prompt:\n\n{user_prompt}\n\n"
                    f"This is the output from the code execution:\n\n{execution_result}\n\n"
                    f"And here's the current content of the webpage:\n\n{webpage_content}\n\n"
                    "Was the task completed successfully?  Provide a brief reason for your answer.")
        ])

        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
        completion_chain = prompt_template | llm | StrOutputParser()
        completion_response = completion_chain.invoke({"input": ""})

        task_complete = "yes" in completion_response.lower() 
        reason = completion_response

        return TaskCompletionOutput(task_complete=task_complete, reason=reason)


class WebAutomationWorkflow:
    """Manages the LangGraph workflow for web automation."""

    def __init__(self, selenium_code_interpreter, selenium_scraping_tool, task_completion_tool):
        self.selenium_code_interpreter = selenium_code_interpreter
        self.selenium_scraping_tool = selenium_scraping_tool
        self.task_completion_tool = task_completion_tool
        self.workflow = self._build_workflow()
        self.compiled_workflow = self.workflow.compile()

    def _build_workflow(self) -> StateGraph:
        """Constructs the LangGraph workflow using LCEL."""
        workflow = StateGraph(LangGraphState)

        load_page_chain = (
            RunnablePassthrough.assign(user_prompt="{input}")  
            | RunnableLambda(lambda x: qa.run(x["user_prompt"]))
            | RunnablePassthrough.assign(webpage_data=lambda x: x)  
        )
        workflow.add_node("load_page", load_page_chain)
        workflow.add_edge(START, "load_page")

        generate_xpath_chain = RunnableLambda(lambda x: generate_xpath_expression(x)) 
        workflow.add_node("generate_xpath", generate_xpath_chain)
        workflow.add_edge("load_page", "generate_xpath")

        generate_selenium_code_chain = RunnableLambda(lambda x: generate_code_from_xpath(x["generated_xpath"], x["user_prompt"]))
        workflow.add_node("generate_selenium_code", generate_selenium_code_chain)
        workflow.add_edge("generate_xpath", "generate_selenium_code")

        execute_code_chain = (
            RunnableLambda.assign(code=lambda x: x["generated_code"]) 
            | self.selenium_code_interpreter
            | RunnablePassthrough.assign(execution_result=lambda x: x) 
        )
        workflow.add_node("execute_code", execute_code_chain)
        workflow.add_edge("generate_selenium_code", "execute_code")

        def completion_check(state: LangGraphState) -> str:
            """Check if the automation task is complete."""
            completion_result = self.task_completion_tool.run(
                user_prompt=state["user_prompt"], 
                execution_result=state["execution_result"],
                webpage_content=state['webpage_data']
            ) 
            if completion_result['task_complete']:
                state["task_complete"] = True
                return END
            else:
                return "load_page"  

        completion_check_chain = RunnableLambda(lambda x: completion_check(x))
        workflow.add_node("completion_check", completion_check_chain)
        workflow.add_edge("execute_code", "completion_check")

        def handle_code_error(state: LangGraphState) -> str:
            """
            Handles errors during Selenium code execution.

            This function analyzes the execution_result and provides 
            feedback to the XPath generation node. 
            """
            error_message = state["execution_result"]

            prompt_template = ChatPromptTemplate.from_messages([
                ("system", "You are a web automation expert, helping to refine XPaths to fix Selenium errors."),
                ("user", f"The following Selenium code execution resulted in an error:\n\n{error_message}\n\n"
                        f"The original user prompt was: \n\n{state['user_prompt']}\n\n"
                        f"The current XPath is: \n\n{state['generated_xpath']}\n\n"
                        "Suggest a refined XPath expression that might fix this error. "
                        "If you can't determine a better XPath, say 'NO SUGGESTIONS'.")
            ])

            llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
            xpath_refinement_chain = prompt_template | llm | StrOutputParser()
            new_xpath_suggestion = xpath_refinement_chain.invoke({"input": ""})

            if new_xpath_suggestion.strip() != "NO SUGGESTIONS":
                state["generated_xpath"] = new_xpath_suggestion.strip() 
            else:
                print("LLM could not provide XPath suggestions for error:", error_message) 

            return "generate_xpath"

        workflow.add_conditional_edges(
            "execute_code", 
            lambda state: "error" in state["execution_result"], 
            {True: handle_code_error}
        )

        return workflow

    def run(self, user_prompt: str) -> LangGraphState:
        """Executes the LangGraph workflow."""
        initial_state = LangGraphState()
        config = {"configurable": {"user_prompt": user_prompt}}
        final_state = self.compiled_workflow.invoke(initial_state, config=config)
        return final_state