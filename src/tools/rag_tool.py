from crewai_tools import BaseTool
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain_groq import ChatGroq

class RAGTool(BaseTool):
    """
    A tool for retrieving relevant webpage information from a Chroma 
    vector database using a RetrievalQA chain.
    """
    name = "Webpage Information Retriever"
    description = (
        "Useful for getting information about a webpage. "
        "Input should be a query related to the content of the webpage."
    )

    def __init__(self, persist_directory: str = "./data"):
        """
        Initializes the RAGTool.

        Args:
            persist_directory (str, optional): The directory where the Chroma 
                database is stored. Defaults to "./data".
        """
        super().__init__()
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.vectordb = Chroma(embedding_function=self.embeddings, persist_directory=persist_directory)
        self.retriever = self.vectordb.as_retriever()
        self.qa_chain = create_retrieval_chain(ChatGroq(model="llama-3.1-8b-instant", temperature=0), self.retriever, chain_type="stuff")

    def _run(self, query: str) -> str:
        """
        Retrieves relevant information from the Chroma vector database.

        Args:
            query (str): The user's query related to the webpage.

        Returns:
            str: The retrieved and synthesized information from the database. 
        """
        return self.qa_chain.run(query)