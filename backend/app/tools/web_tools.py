# from langchain.tools import tool
# from app.services.web_search_service import web_search_service


# @tool("web_search")
# def web_search(query: str) -> str:
#     """
#     To be used only when local documents do not contain the answer or when
#     the user explicitly asks for current external information.
#     """
#     results = web_search_service.search(query)
#     return results if results else "No relevant web results were found."