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
from pymongo.collection import Collection
from router import run_workflow
from langchain_mongodb import MongoDBAtlasVectorSearch
from bson import ObjectId

load_dotenv()

# ---------------------------------------------------------
# 1️⃣  MONGODB CLIENT (procedure collection)
# ---------------------------------------------------------

MONGODB_ATLAS_URI = os.getenv("MONGODB_ATLAS_URI")
client = MongoClient(MONGODB_ATLAS_URI)

db_name = "rag_bot"
collection_name = "embeddings"

def mongodb_client():
   return client[db_name][collection_name]


embeddings = AzureOpenAIEmbeddings(
    model="text-embedding-ada-002",
    api_key=os.environ.get("AZURE_OPENAI_KEY"),
    azure_endpoint=os.environ.get("ENDPOINT_URL")
)



db_name = "rag_bot"

def mongodb_procedure_client():
    return client[db_name]["procedure"]

collection_procedure = mongodb_procedure_client()

# procedure_json_extraction 


vector_store_procedure = MongoDBAtlasVectorSearch(
    collection=collection_procedure,
    embedding=embeddings,
    index_name="default", 
    text_key="page_content",
    relevance_score_fn="cosine"
)

def detect_procedure(user_input: str, tenant_id: str):
    retriever = vector_store_procedure.as_retriever(
        search_kwargs={
            "pre_filter": {
                "metadata.tenant_id": tenant_id,
            },
            "k": 1
        }
    )
    # 🔑 THIS is the retrieval result
    documents = retriever.invoke(user_input)

    print("Retrieved documents:", documents)

    # ✅ Proper empty check
    if not documents:
        return None

    doc = documents[0]
    print("Document metadata:", doc.metadata)

    # ✅ Safe metadata access
    procedure_id = doc.metadata.get("metadata", {}).get("procedure_id")
    print("Detected procedure ID:", procedure_id)
    if not procedure_id:
        return None

    return procedure_id
# procedure_json fetch by id

def mongodb_procedure_json_client():
    return client[db_name]["procedure_json"]

collection_procedure_json = mongodb_procedure_json_client()

def get_procedure_by_id(
    collection: Collection,
    procedure_id: str,
    tenant_id: str
):
    doc = collection.find_one(
        {
            "metadata.procedure_id": procedure_id,
            "metadata.tenant_id": tenant_id
        } # optional: hide Mongo ObjectId
    )

    if not doc:
        raise ValueError("Procedure not found")

    return doc
# Example usage

