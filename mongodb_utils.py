# mongodb_utils.py
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredHTMLLoader, BSHTMLLoader
from bs4 import BeautifulSoup
from langchain.docstore.document import Document
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from langchain_community.document_transformers import BeautifulSoupTransformer
from pymongo import MongoClient
from typing import Dict
import os
import logging
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------
# 1️⃣  MONGODB CLIENT (procedure collection)
# ---------------------------------------------------------

MONGODB_ATLAS_URI = os.getenv("MONGODB_ATLAS_URI")
client = MongoClient(MONGODB_ATLAS_URI)

db_name = "rag_bot"

def mongodb_procedure_client():
    return client[db_name]["procedure"]

collection_procedure = mongodb_procedure_client()

# ---------------------------------------------------------
# 2️⃣  Embedding Model
# ---------------------------------------------------------

embeddings = AzureOpenAIEmbeddings(
    model="text-embedding-ada-002",
    api_key=os.environ.get("AZURE_OPENAI_KEY"),
    azure_endpoint=os.environ.get("ENDPOINT_URL")
)

