from crewai import Crew, Process
from langchain_groq import ChatGroq
from agents.web_automation_specialist import WebAutomationSpecialist

class WebAutomationCrew(Crew):
    """
    A CrewAI crew for web automation tasks.

    This crew consists of a single agent, the Web Automation Specialist,
    which is responsible for executing tasks related to automating actions
    on webpages.  
    """

    def __init__(self):
        super().__init__(
            agents=[WebAutomationSpecialist()],
            tasks=[],
            process=Process.sequential,
            memory=True,
            verbose=True,
            planning=True,
            planning_llm=ChatGroq(model="llama-3.1-8b-instant")
        )