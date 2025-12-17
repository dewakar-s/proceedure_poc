from mongodb_utils import collection_procedure
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain.tools import StructuredTool
from mongodb_utils import  embeddings, collection_procedure 
from langchain_openai import AzureChatOpenAI
import os
import re
import json
from router import run_workflow

AZURE_API_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_ENDPOINT = os.getenv("ENDPOINT_URL")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")
API_VERSION = "2025-03-01-preview"

vector_store_procedure = MongoDBAtlasVectorSearch(
    collection= collection_procedure,
    embedding= embeddings,
    index_name="default",
    relevance_score_fn="cosine"

)

try:
    llm = AzureChatOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        openai_api_key=AZURE_API_KEY,
        azure_deployment=DEPLOYMENT_NAME,
        openai_api_version=API_VERSION,
        temperature=0.7,
    )
    print("✅ AzureChatOpenAI initialized")
except Exception as e:
    print("❌ Error:", e)
    exit()


def create_retriever_procedure(tenant_id):
    return vector_store_procedure.as_retriever(
        search_kwargs={
            "pre_filter": {
                "metadata.tenant_id": tenant_id   
            },
            "k": 1
        }
    )
def generate_plan(user_query,tenant_id):
    # 1. Retrieve matching procedure
    retriever = create_retriever_procedure(tenant_id)
    print(retriever.invoke(user_query))
    docs = retriever.get_relevant_documents(user_query)
    procedure_text = docs[0].page_content

    # 2. Send to LLM along with user query
    planning_prompt = f"""
You are an expert at creating step-by-step execution plans based on user requests and predefined procedures.
The execution plan should contain all steps mentioned in the data each step should contain three fields: type, action, and message/action_id as applicable.
And the produce text is 
{procedure_text}
The output should be in json format 
IT Must contain tyoe API_CALL , ASK_USER , RESPOND_FINAL
Second what kind of action it is doing
third should action_id present or else message to be shown to user
Based on the above procedure, generate a detailed execution plan 
do not give any explanation just provide json output
there should be steps[] inside this array there should be multiple steps
    """
    response = llm.invoke(planning_prompt)  # LLM CALL ONCE
    return response.content


answer = generate_plan("i want to create a user ", "a9a5bcdc-d607-4fc6-a0a0-2469b383af6b")

def extract_json(text: str):
    text = re.sub(r"```json|```", "", text).strip()
    return json.loads(text)

print("Generated Execution Plan:", answer)
the_json = extract_json(answer)
print(the_json)
run_workflow(the_json)