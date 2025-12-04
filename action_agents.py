from langchain_openai import AzureChatOpenAI
import os
import dotenv
from langchain.agents import create_react_agent
from langchain.prompts import PromptTemplate


dotenv.load_dotenv()


AZURE_API_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_ENDPOINT = os.getenv("ENDPOINT_URL")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")
API_VERSION = "2025-03-01-preview"


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

