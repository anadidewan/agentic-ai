from typing import Any


class AgentTraceService:
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.tool_calls: list[dict[str, Any]] = []
        self.retrieved_chunks: list[dict[str, Any]] = []
        self.sources: list[dict[str, Any]] = []

    def add_tool_call(self, tool_name: str, input_data: dict[str, Any] | None = None) -> None:
        self.tool_calls.append({
            "tool_name": tool_name,
            "input": input_data or {},
        })

    def set_retrieved_chunks(self, chunks: list[dict[str, Any]]) -> None:
        self.retrieved_chunks = chunks

    def set_sources_from_chunks(self, chunks: list[dict[str, Any]]) -> None:
        seen = set()
        sources = []

        for chunk in chunks:
            key = (chunk.get("document_name"), chunk.get("chunk_id"))
            if key in seen:
                continue
            seen.add(key)
            sources.append({
                "document_name": chunk.get("document_name"),
                "chunk_id": chunk.get("chunk_id"),
            })

        self.sources = sources

    def snapshot(self) -> dict[str, Any]:
        return {
            "tool_calls": self.tool_calls,
            "retrieved_chunks": self.retrieved_chunks,
            "sources": self.sources,
        }


agent_trace_service = AgentTraceService()