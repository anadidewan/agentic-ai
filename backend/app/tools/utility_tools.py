from langchain.tools import tool


@tool("summarize_context")
def summarize_context(context: str, max_points: int = 5) -> str:
    """
    Compress long retrieved content into a concise summary.

    To be when retrieved text is too long or repetitive and should be
    condensed before final answer generation.
    """
    lines = [line.strip() for line in context.splitlines() if line.strip()]
    condensed = lines[:max_points]
    if not condensed:
        return "No content provided to summarize."
    return "\n".join(f"- {line}" for line in condensed)


@tool("calculator")
def calculator(expression: str) -> str:
    """
    Evaluate a basic arithmetic expression.

    to be used for numeric reasoning such as growth rate, deltas, percentages,
    totals, and averages.
    """
    try:
        allowed_names = {}
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Calculation error: {str(e)}"