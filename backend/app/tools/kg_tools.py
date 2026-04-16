from langchain.tools import tool
from app.services.retrieval_service import hybrid_retrieve, graph_expand, get_graph_context
from app.services.agent_trace_service import agent_trace_service


@tool("knowledge_graph_lookup")
def knowledge_graph_lookup(query: str, top_k: int = 6) -> str:
    """
   Implementing graph retrieval
    """
    agent_trace_service.add_tool_call(
        tool_name="knowledge_graph_lookup",
        input_data={"query": query, "top_k": top_k},
    )
    try:
        retrieved = hybrid_retrieve(query, top_k=top_k)
        expanded = graph_expand(retrieved, top_k=3)
        graph_context = get_graph_context(expanded)

        if not expanded:
            return "No relevant graph-backed results were found."
        
        agent_trace_service.set_retrieved_chunks(expanded)
        agent_trace_service.set_sources_from_chunks(expanded)

        context_parts = []
        for i, chunk in enumerate(expanded[:5], start=1):
            matched_entities = chunk.get("matched_entities", [])
            matched_str = ", ".join(matched_entities) if matched_entities else "None"

            context_parts.append(
                f"[Expanded Chunk {i}]\n"
                f"Document: {chunk['document_name']}\n"
                f"Chunk ID: {chunk['chunk_id']}\n"
                f"Matched Entities: {matched_str}\n"
                f"Text: {chunk['text']}\n"
            )

        graph_section = f"\nRelevant Graph Relationships:\n{graph_context}" if graph_context else ""
        return "\n".join(context_parts) + graph_section

    except Exception as e:
        return f"Knowledge graph lookup failed: {str(e)}"