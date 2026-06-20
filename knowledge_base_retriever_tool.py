
import logging
from langchain.docstore.document import Document
from langchain.vectorstores.base import VectorStore
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain.tools.retriever import create_retriever_tool
from mongodb_uttilies_procedure import embeddings, mongodb_client

import time

collection = mongodb_client()

# implementing embedding Features
vector_store = MongoDBAtlasVectorSearch(
    collection=collection,
    embedding=embeddings,
    index_name="vector_index", 
    relevance_score_fn="cosine"
)

logger = logging.getLogger(__name__)

def create_retriever(tenant_id, retriever_mode):
    base_filter = {"metadata.tenant_id": tenant_id}
    
    if retriever_mode == "agent_access":

         pre_filter_query = {**base_filter, "metadata.agent_access": True}

    elif retriever_mode == "copilot_access":

         pre_filter_query = {**base_filter, "metadata.copilot_access": True}
        
    else:
        logger.warning(f"Unknown retriever mode: {retriever_mode}. Falling back to tenant_id filter only.")
        pre_filter_query = base_filter

    # --- Create the retriever with the combined pre_filter ---
    retriever = vector_store.as_retriever(
        search_kwargs={
            "pre_filter": pre_filter_query,
            "k": 5
        }
    )

    # Wrap retriever with filtering logic
    class FilteredRetriever:
        def __init__(self, retriever, score_threshold=0.75):
            self.retriever = retriever
            self.score_threshold = score_threshold

        # Use .invoke instead of get_relevant_documents
        def invoke(self, query: str, **kwargs):
            start = time.perf_counter()
            docs = self.retriever.invoke(query, **kwargs)
            filtered_docs = []

            for d in docs:
                if d.metadata.get("score", 1.0) >= self.score_threshold:
                    # Handle nested metadata safely
                    nested_metadata = d.metadata.get("metadata", {})
                    url = nested_metadata.get("url") or d.metadata.get("url") or "N/A"

                    # Format page content to include URL context
                    formatted_page = f"{d.page_content}\n(Source: {url})"

                    # Return a clean Document with formatted text and top-level metadata
                    new_doc = Document(
                        page_content=formatted_page,
                        metadata={
                            "source": url,
                            "url": url,
                            "score": d.metadata.get("score")
                        }
                    )
                    filtered_docs.append(new_doc)
            end = time.perf_counter()
            logging.info(f"FilteredRetriever.invoke time: {end - start} seconds")
            print(f"[PERF] FilteredRetriever execution time: {end - start:.3f}s")
            return filtered_docs

    filtered_retriever = FilteredRetriever(retriever)
    return create_retriever_tool(
        retriever=filtered_retriever,
        name="document_retriever",
        description="Use this tool to fetch relevant documents."
        )

