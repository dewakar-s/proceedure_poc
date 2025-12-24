import os
import re
import json
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_openai import AzureChatOpenAI

from mongodb_uttilies_procedure import embeddings, collection_procedure

# -------------------- CONFIG --------------------
AZURE_API_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_ENDPOINT = os.getenv("ENDPOINT_URL")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")
API_VERSION = "2025-03-01-preview"

# -------------------- VECTOR STORE --------------------
vector_store_procedure = MongoDBAtlasVectorSearch(
    collection=collection_procedure,
    embedding=embeddings,
    index_name="default",
    relevance_score_fn="cosine",
)

# -------------------- LLM --------------------
llm = AzureChatOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    openai_api_key=AZURE_API_KEY,
    azure_deployment=DEPLOYMENT_NAME,
    openai_api_version=API_VERSION,
    temperature=0.7,
)

# -------------------- REQUEST MODEL --------------------
class PlanRequest(BaseModel):
    user_query: str

# -------------------- HELPERS --------------------
def create_retriever_procedure(tenant_id: str):
    return vector_store_procedure.as_retriever(
        search_kwargs={
            "pre_filter": {"metadata.tenant_id": tenant_id},
            "k": 1,
        }
    )


def extract_json(text: str):
    text = re.sub(r"```json|```", "", text).strip()
    return json.loads(text)


def generate_plan(user_query: str, tenant_id: str):
    retriever = create_retriever_procedure(tenant_id)
    docs = retriever.get_relevant_documents(user_query)

    if not docs:
        raise HTTPException(status_code=404, detail="No procedure found")

    procedure_text = docs[0].page_content

    planning_prompt = f"""
You are an expert at creating step-by-step execution plans based on user requests and predefined procedures.

The procedure text is:
{procedure_text}

Rules:
- Output MUST be valid JSON
- Must contain steps[]
- Each step must include:
  - type (API_CALL | ASK_USER | RESPOND_FINAL)
  - action
  - action_id OR message

Return only JSON. No explanation.
"""

    response = llm.invoke(planning_prompt)
    return extract_json(response.content)

# -------------------- ENDPOINT --------------------
