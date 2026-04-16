from typing import Optional
from langchain.tools import tool
from app.services.retrieval_service import hybrid_retrieve
from app.services.agent_trace_service import agent_trace_service



@tool("document_search")
def document_search(query: str, top_k: int = 5, document_name: Optional[str] = None) -> str:
    """
    To be used when question reply is from local files or previosuly fetched files. This is the first tool to be used for any question related to documents. It uses a hybrid search approach that combines vector similarity with keyword matching to find relevant chunks of text from the local document collection.
    Args:
        query: Natural-language search query.
        top_k: Number of chunks to retrieve.
        document_name: Optional document filter when the user names a specific document.
    """
    agent_trace_service.add_tool_call(
        tool_name="document_search",
        input_data={"query": query, "top_k": top_k},
    )
    try:
        results = hybrid_retrieve(query, top_k=top_k)
    except Exception as e:
        return f"Document search failed: {str(e)}"
    

    if not results:
        return "No relevant document chunks were found."
    
    agent_trace_service.set_retrieved_chunks(results)
    agent_trace_service.set_sources_from_chunks(results)
    
    formatted = []
    for i, chunk in enumerate(results, start=1):
        formatted.append(
            f"[Chunk {i}]\n"
            f"Document: {chunk['document_name']}\n"
            f"Chunk ID: {chunk['chunk_id']}\n"
            f"Hybrid Score: {round(chunk.get('hybrid_score', 0.0), 4)}\n"
            f"Text: {chunk['text']}\n"
        )

    return "\n".join(formatted)