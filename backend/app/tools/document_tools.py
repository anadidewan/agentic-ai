from typing import Optional
from langchain.tools import tool
from app.services.retrieval_service import retrieval_service


@tool("document_search")
def document_search(query: str, top_k: int = 5, document_name: Optional[str] = None) -> str:
    """
    To be used when question reply is from local files or previosuly fetched files. This is the first tool to be used for any question related to documents. It uses a hybrid search approach that combines vector similarity with keyword matching to find relevant chunks of text from the local document collection.
    Args:
        query: Natural-language search query.
        top_k: Number of chunks to retrieve.
        document_name: Optional document filter when the user names a specific document.
    """
    try:
        results = retrieval_service.hybrid_retrieve(query, top_k=top_k)
    except Exception as e:
        return f"Document search failed: {str(e)}"

    if not results:
        return "No relevant document chunks were found."

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