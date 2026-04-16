from langchain.tools import tool
from app.services.kg_service import kg_service


@tool("knowledge_graph_lookup")
def knowledge_graph_lookup(query: str) -> str:
    """
   Implementing graph retrieval
    """
    result = kg_service.lookup(query)
    return result if result else "No useful knowledge graph results were found."