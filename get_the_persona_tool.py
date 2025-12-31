import requests
from langchain.tools import StructuredTool
import os
import time
import logging

persona_url = os.environ.get("PERSONA_URL")

def create_persona_tool(tenant_id: str):
    def _get_persona():
        start_time = time.perf_counter()
        try:
            headers = {"X-Tenant-Id": tenant_id}
            response = requests.get(persona_url, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()

            persona_data = data.get("data", {})
            end_time = time.perf_counter()
            logging.info(f"get_persona_tool execution time: {end_time - start_time} seconds")
            print(f"[PERF] get_persona_tool execution time: {end_time - start_time:.3f}s")
            return {
                "name": persona_data.get("name") or "AI Assistant",
                "description": persona_data.get("description", ""),
                "greetingMessage": persona_data.get("greetingMessage") or "Hello! How can I assist you today?",
                "temperature": persona_data.get("temperature"),
                "maxTokens": persona_data.get("maxTokens"),
                "messageLength": persona_data.get("messageLength") or "medium",
                "tone": persona_data.get("tone") or "professional",
                "allowEmoji": persona_data.get("allowEmoji", False),
            }
        except Exception as e:
            return {"error": f"Failed to fetch persona: {str(e)}"}

    return StructuredTool.from_function(
        func=_get_persona,
        name="get_persona_tool",
        description="Fetches persona information (name, description, greetingMessage, tone, messageLength, allowEmoji, etc.) for the current tenant"
    )

