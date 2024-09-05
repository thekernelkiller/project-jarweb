from crewai import Task
from textwrap import dedent

class WebAutomationTasks:
    """Defines tasks for web automation."""
    def web_automation_task(self, agent, user_prompt: str) -> Task:
        """
        A task for automating actions on a webpage based on a user prompt. 

        The prompt should include:
        - The URL of the target webpage.
        - A clear and concise description of the actions to be performed. 
        
        For example:
        "https://www.example.com Find the search bar, type 'LangChain', and click the search button."
        """
        return Task(
            description=dedent(f"""\
                You are a web automation expert. 
                Your task is to perform the following actions on a webpage:

                {user_prompt}

                Provide a detailed report of the outcome of your automation.
                If you encounter errors, describe them clearly and attempt 
                to identify the cause of the error.
                """),
            agent=agent,
            expected_output="A report of the web automation process and outcome." 
        )