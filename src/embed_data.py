import os
import json
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from tqdm import tqdm
import asyncio

BATCH_SIZE = 500

def load_json(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, 'r') as file:
        data = json.load(file)
    return [{"content": json.dumps(data), "source": file_path}]

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    documents = []
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            documents.append({"content": json.dumps(data), "source": file_path})
    return documents

def embed_batch(texts_batch, embeddings, persist_directory, batch_number):
    print(f"Embedding batch {batch_number} with {len(texts_batch)} texts...")
    vectordb = Chroma.from_documents(texts_batch, embeddings, persist_directory=persist_directory)
    vectordb.persist()
    print(f"Batch {batch_number} embedded successfully!")

def embed_website_data(data_directory: str, persist_directory: str = "./data", batch_size: int = BATCH_SIZE):
    documents = []

    loaders = [
        {"loader": DirectoryLoader(os.path.join(data_directory), glob="**/*.json", loader_cls=lambda path: TextLoader(path, encoding="utf-8")), "type": "JSON"},
        {"loader": DirectoryLoader(os.path.join(data_directory), glob="**/*.jsonl", loader_cls=lambda path: TextLoader(path, encoding="utf-8")), "type": "JSONL"},
        {"loader": DirectoryLoader(os.path.join(data_directory, "pages"), glob="**/*.html", loader_cls=lambda path: TextLoader(path, encoding="utf-8")), "type": "HTML"}
    ]

    for loader_info in loaders:
        loader = loader_info["loader"]
        loader_type = loader_info["type"]
        print(f"Loading {loader_type} files...")

        try:
            for doc in tqdm(loader.load(), desc=f"Processing {loader_type} files"):
                documents.append(doc)
        except Exception as e:
            print(f"Error loading {loader_type} files: {str(e)}")

    if not documents:
        print(f"No compatible files found in directory: {data_directory}")
        return

    print(f"Total documents loaded: {len(documents)}")

    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    print(f"Total chunks created: {len(texts)}")

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    for i in range(0, len(texts), batch_size):
        texts_batch = texts[i:i + batch_size]
        embed_batch(texts_batch, embeddings, persist_directory, i // batch_size + 1)

    print(f"All batches processed successfully!")

async def async_embed_batch(texts_batch, embeddings, persist_directory, batch_number):
    print(f"Embedding batch {batch_number} with {len(texts_batch)} texts asynchronously...")
    vectordb = Chroma.from_documents(texts_batch, embeddings, persist_directory=persist_directory)
    vectordb.persist()
    print(f"Batch {batch_number} embedded successfully!")

async def embed_website_data_async(data_directory: str, persist_directory: str = "./data", batch_size: int = BATCH_SIZE):
    documents = []

    loaders = [
        {"loader": DirectoryLoader(os.path.join(data_directory), glob="**/*.json", loader_cls=lambda path: TextLoader(path, encoding="utf-8")), "type": "JSON"},
        {"loader": DirectoryLoader(os.path.join(data_directory), glob="**/*.jsonl", loader_cls=lambda path: TextLoader(path, encoding="utf-8")), "type": "JSONL"},
        {"loader": DirectoryLoader(os.path.join(data_directory, "pages"), glob="**/*.html", loader_cls=lambda path: TextLoader(path, encoding="utf-8")), "type": "HTML"}
    ]

    for loader_info in loaders:
        loader = loader_info["loader"]
        loader_type = loader_info["type"]
        print(f"Loading {loader_type} files...")

        try:
            for doc in tqdm(loader.load(), desc=f"Processing {loader_type} files"):
                documents.append(doc)
        except Exception as e:
            print(f"Error loading {loader_type} files: {str(e)}")

    if not documents:
        print(f"No compatible files found in directory: {data_directory}")
        return

    print(f"Total documents loaded: {len(documents)}")

    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    print(f"Total chunks created: {len(texts)}")

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    tasks = []
    for i in range(0, len(texts), batch_size):
        texts_batch = texts[i:i + batch_size]
        tasks.append(async_embed_batch(texts_batch, embeddings, persist_directory, i // batch_size + 1))

    await asyncio.gather(*tasks)

    print(f"All batches processed successfully!")

if __name__ == "__main__":
    website_data_dir = "./src/website_data"
    
    # For synchronous batched embedding:
    embed_website_data(website_data_dir)
    
    # For asynchronous batched embedding:
    # asyncio.run(embed_website_data_async(website_data_dir))
