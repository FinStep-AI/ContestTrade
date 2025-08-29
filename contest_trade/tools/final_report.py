from langchain_core.tools import tool
from pydantic import BaseModel, Field

class FinalReportInput(BaseModel):
    report_content: str = Field(description="The final report content to be submitted")

@tool(
    description="""Generate a final report to the user. The task can't continue when your call this tool. So make sure you have enough information to write a report.""",
    args_schema=FinalReportInput
)
async def final_report(report_content: str):
    """生成最终报告并标记任务完成"""
    print(f"📊 Final report generated with {len(report_content)} characters")
    return f"Final report submitted successfully. Content length: {len(report_content)} characters."
