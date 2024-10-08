# Say Hello to JarWeb! 👋

JarWeb is a web automation system built using CrewAI and LangGraph.  It takes a human-like prompt with your instructions and then dynamically figures out how to automate actions on any website. Think of it as your own personal Jarvis for the web!

Read the detailed blog of my approach to this project [here](https://haberdashery.vivekirl.space/private/interface-labs-assignment).

> Note: This project is still in WIP, I spent a lot of time on designing the solution than the implementation. I will complete this project very soon. Regardless, I'm ready for feedback and discussions!

### 1. Key Features

- **Dynamic Reasoning:** JarWeb uses LLMs (large language models) to understand your instructions and the structure of the webpage, enabling it to adapt to various scenarios and websites.
- **Context Grounding (RAG):** A robust RAG (Retrieval Augmented Generation) system ensures the agent has accurate and relevant information about the target website.
- **Selenium Automation:**  JarWeb leverages Selenium to perform actions like clicking buttons, filling in forms, and extracting data. 
- **Code Interpreter:** A secure code interpreter executes Selenium code, ensuring safe and reliable automation. 
- **Human-in-the-Loop (Planned):**  JarWeb is designed for future integration of human oversight, allowing you to review and approve actions before execution or provide feedback to improve the agent's performance.
  
### 2. Architecture

<img width="949" alt="Screenshot 2024-09-05 at 12 40 00" src="https://github.com/user-attachments/assets/cdac58ac-91d4-485b-b3a5-127099e5fbae">

### 3. How to run

1. Make sure you have `conda` installed.
2. Create a new virtual environment using `conda create --name <venv-name>`.
3. Activate the venv: `conda activate <venv-name>`.
4. Run `pip install -r requirements.txt`.
5. Make sure you have [Ollama](https://ollama.com/download) installed and running, because we're using the `nomic-embed-text` model for embedding the website data to the chromadb.
6. Run `python src/embed_data.py` for creating the embeddings.
7. Change `.env_example` file name to `.env` and get Groq API key from [Groq Cloud Console](https://console.groq.com/). 
8. Finally, run `python src/main.py`. 


