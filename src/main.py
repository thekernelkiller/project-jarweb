from agents.web_automation_specialist import WebAutomationSpecialist
from tasks.tasks import WebAutomationTasks 
from crew import WebAutomationCrew

def main():
    """
    Main execution function for the web automation agent.
    """
    user_prompt = input("Please describe the web automation task you'd like to perform: ")

    web_automation_tasks = WebAutomationTasks()
    web_automation_task = web_automation_tasks.web_automation_task(
        agent=WebAutomationSpecialist(), user_prompt=user_prompt
    )

    web_automation_crew = WebAutomationCrew()
    web_automation_crew.add_task(web_automation_task)

    final_state = web_automation_crew.kickoff()

    print("\n\n*** Web Automation Report ***\n") 

    print(f"User Prompt: {user_prompt}")
    print(f"Generated XPath: {final_state['generated_xpath']}")
    print(f"Generated Code:\n{final_state['generated_code']}")
    print(f"Execution Result:\n{final_state['execution_result']}")
    print(f"Task Complete: {final_state['task_complete']}")

if __name__ == "__main__":
    main()