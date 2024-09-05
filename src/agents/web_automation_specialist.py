from typing import Dict, Any, List, Optional
from crewai import Agent, Task
from crewai_tools import CodeInterpreterTool, SeleniumScrapingTool
from langgraph.workflow import WebAutomationWorkflow, qa 
from langgraph.workflow import TaskCompletionTool

class WebAutomationSpecialist(Agent):
    """
    An AI agent specialized in web automation tasks using Selenium.
    
    This agent leverages a LangGraph workflow to manage the web automation 
    process, including: 
        - Retrieving relevant webpage data using a RAG system (Chroma).
        - Extracting user instructions from prompts.
        - Generating dynamic XPaths to target webpage elements.
        - Creating Selenium code to automate actions.
        - Executing the code using a code interpreter.
        - Handling errors and checking for task completion.
    """
    def __init__(self):
        super().__init__()
        self.role = "Web Automation Specialist"
        self.goal = (
            "To accurately and efficiently automate tasks on webpages "
            "according to user instructions."
        )
        self.backstory = (
            "I am an expert in web automation, proficient in understanding "
            "user requests, extracting data from webpages, interacting with "
            "elements, and handling dynamic content. I leverage tools for " 
            "context understanding and Selenium for precise web actions. "
            "My primary objective is to successfully complete user-requested "
            "automation tasks with a high degree of reliability." 
        )
        self.tools = [ 
            qa,
            CodeInterpreterTool(),
            SeleniumScrapingTool(),
            TaskCompletionTool() 
        ]
        self.verbose = True 
        self.allow_delegation = False  

    def execute(self, task: Task) -> Dict[str, Any]: 
        """Executes the web automation task using the LangGraph workflow."""
        workflow = WebAutomationWorkflow(
            selenium_code_interpreter=self.tools[1],
            selenium_scraping_tool=self.tools[2],
            task_completion_tool=self.tools[3]
        )

        user_prompt = task.description 

        final_state = workflow.run(user_prompt=user_prompt)

        return final_state 