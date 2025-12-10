from mongodb_utils import collection_procedure
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain.tools import StructuredTool
from mongodb_utils import  embeddings, collection_procedure 
from langchain_openai import AzureChatOpenAI
import os

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
    You are an **Execution Plan Optimizer**. Your task is to generate the most efficient, structured JSON plan based on the user's request and the Procedure Blueprint.

    1. **Optimization Rule (CRITICAL: Skip and Inject):**
        a. **SKIP:** If the 'User Query' already contains a piece of required information (e.g., email ID, order ID, or any value mentioned in a 'required_info' field of the blueprint), you **MUST OMIT** the corresponding 'ASK_USER' step from the final JSON plan.
        b. **INJECT:** You must then **INJECT** the value directly from the 'User Query' into the 'parameters' of the next subsequent 'API_CALL' step, replacing the placeholder value (e.g., '<user_provided_email>'). You must deduce the parameter name (e.g., 'email_id') from the context.
    
    2. **Required Schema:** The output MUST be a JSON object with a single 'steps' array. Each step must use one of these types: 'API_CALL', 'ASK_USER', or 'RESPOND_FINAL'.
    
    3. **Blueprint Analysis:** You must parse the steps from the 'Procedure Blueprint' and apply the optimization rule.

    ---
    User Query (Analyze this for pre-filled data, e.g., 'dewasrfhh@gmail.com'):
    {user_query}
    ---
    
    Procedure Blueprint (Describes the required sequence of API calls and data gathering, assumed to be in a machine-readable format like the one provided by the user in the past):
    {procedure_text}
    ---
    
    Generate the FINAL, optimized JSON plan below:
    """
    response = llm.invoke(planning_prompt)  # LLM CALL ONCE
    return response.content


answer = generate_plan("i want to cancel my order  ", "021ee120-7cc2-4f78-9ff5-db9e785c0118")
print("Generated Execution Plan:", answer)