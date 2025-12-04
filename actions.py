import logging
from mongodb_utilies import get_actions_collection
from pydantic import BaseModel, Field, create_model
from langchain.tools import StructuredTool
import requests
import re


DEFAULT_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "User-Agent": "DynamicToolAgent/1.0"
}

# -----------------------
# Safe replacement for eval()
# -----------------------
TYPE_MAP = {
    "str": str,
    "string": str,
    "int": int,
    "integer": int,
    "float": float,
    "double": float,
    "number": float,
    "Number": float,
    "bool": bool,
    "boolean": bool,
}

def sanitize_tool_name(name: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_\.-]', '_', name)


def build_tool_from_json(tool_data: dict) -> StructuredTool:
    """Build a StructuredTool compatible with old LangChain."""

    tool_name = sanitize_tool_name(tool_data["name"])
    logging.info(f"Building tool: {tool_name}")

    # ------------------- 1) Build Pydantic schema -------------------
    params_dict = {}

    for p in tool_data.get("parameters", []):
        field_name = p.get("name")
        field_type = p.get("type", "str")

        # SAFE TYPE LOOKUP
        python_type = TYPE_MAP.get(field_type, str)

        params_dict[field_name] = (python_type, Field(...))

    DynamicSchema = create_model(
        f"{tool_data['name']}Input",
        __base__=BaseModel,
        **params_dict
    )

    # ------------------- 2) Headers -------------------
    extra_headers = {}
    for h in tool_data.get("headers", []):
        extra_headers[h["key"]] = h["value"]

    # ------------------- 3) API call function -------------------
    def dynamic_func(**kwargs):
        method = tool_data.get("httpMethod", "GET").upper()
        url_template = tool_data.get("url", "")
        
        # Fix double braces ({{ }}) to single braces ({ })
        url_template = url_template.replace("{{", "{").replace("}}", "}")

        # DEBUG: Print what we received
        #print(f"DEBUG - URL Template: {url_template}")
        #print(f"DEBUG - Kwargs received: {kwargs}")

        headers = {**DEFAULT_HEADERS, **extra_headers}

        # Extract path parameters from URL template
        path_params = re.findall(r'\{(\w+)\}', url_template)
        #print(f"DEBUG - Path params found: {path_params}")
        
        # Separate path params from query params
        path_values = {}
        query_params = {}
        
        for key, value in kwargs.items():
            if key in path_params:
                # Convert floats to ints if they're whole numbers
                if isinstance(value, float) and value.is_integer():
                    value = int(value)
                path_values[key] = value
            else:
                query_params[key] = value
        
        #print(f"DEBUG - Path values: {path_values}")
        #print(f"DEBUG - Query params: {query_params}")
        
        # Format URL with path parameters only
        try:
            url = url_template.format(**path_values)
            #print(f"DEBUG - Final URL: {url}")
        except KeyError as e:
            logging.error("KeyERROR DURING FORMAT {e}.")
            # Fallback: try formatting with all kwargs
            url = url_template
            for key, value in path_values.items():
                url = url.replace(f"{{{key}}}", str(value))
            logging.error(f"Falling url {url}.")

        try:
            if method == "GET":
                resp = requests.get(url, headers=headers, params=query_params if query_params else None)
            elif method == "POST":
                resp = requests.post(url, headers=headers, json=kwargs)
            elif method == "PUT":
                resp = requests.put(url, headers=headers, json=kwargs)
            elif method == "DELETE":
                resp = requests.delete(url, headers=headers, json=kwargs)
            else:
                return {
                    "status": "failed",
                    "message": f"Unsupported HTTP method: {method}",
                    "data": []
                }

            resp.raise_for_status()

            return {
                "status": "success",
                "message": f"Success {kwargs}",
                "data": resp.json() if resp.content else {}
            }

        except Exception as e:
            return {
                "status": "failed",
                "message": str(e),
                "data": []
            }

    # ------------------- 4) Build final tool object -------------------
    return StructuredTool(
        name=tool_name,
        description=tool_data.get("description", ""),
        func=dynamic_func,
        args_schema=DynamicSchema
    )


def create_tool_retriever(action_id):
    # Fetch actions from DB (cursor)
    cursor = get_actions_collection(action_id)

    # Convert cursor → list
    tool_docs = list(cursor)
    logging.info(f"Tools fetched from DB: {len(tool_docs)}")

    tools = []
    for doc in tool_docs:
        tool = build_tool_from_json(doc)
        if tool:
            tools.append(tool)

    return tools

def action(action_id):
    print("This is an action function.")
    tools_builder = create_tool_retriever(action_id)
    print(f"Number of tools created: {len(tools_builder)}")