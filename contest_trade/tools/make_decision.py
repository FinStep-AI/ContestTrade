from langchain_core.tools import tool

@tool(
    description="Make a decision under current given informations. You must make a decision when the budget is not enough to fetch more information.",
)
async def make_decision():
    print("make_decision")
    return None
