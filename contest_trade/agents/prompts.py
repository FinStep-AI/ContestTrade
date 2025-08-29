prompt_for_research_plan = """
<Task>
{task}
</Task>

<Current_Time>
{current_time}
</Current_Time>

<Background_Information>
{background_information}
</Background_Information>

<Available_Tools>
{tools_info}
</Available_Tools>

Please create a detailed step-by-step plan to complete the following user task based on the background information provided.
Each step should be a clear, actionable instruction. Each step should be a one-line description explaining what needs to be done.
Steps should:
1. Be specific and actionable
2. Use appropriate tools from those provided
3. Be arranged in logical order
4. Consider contextual information
5. Return a list of strings, where each element is a step, without unnecessary words or explanations
6. Focus on information gathering or visualization, no analysis or summary steps needed
7. Not exceed 5 steps
8. Output result in language: {output_language}

Please output the action plan in the following format, do not output any other information:
1. xxx
2. xxx
"""

prompt_for_research_choose_tool = """
<Task>
{task}
</Task>

<Current_Time>
{current_time}
</Current_Time>

<Background_Information>
{background_information}
</Background_Information>

<Your_Plan>
{plan}
</Your_Plan>

<Available_Tools>
{tools_info}
</Available_Tools>

<Current_Task_Context>
{tool_call_context}
</Current_Task_Context>

Analyze the following steps and select tools:
## Available Resources
You currently have access to the following analysis tools:
- **Financial Data Tools**: Get company financials, market data, and historical information
- **News & Information Tools**: Search for recent news, announcements, and market updates
- **Research Tools**: Access analyst reports, industry data, and comparative analysis
- **Web Search Tools**: Get real-time information from various sources

## Development Needs Assessment
If you encounter limitations in your analytical capabilities that prevent you from completing high-quality research, you can propose new tool development. Consider whether you need specialized tools for:
- Advanced technical analysis and charting capabilities
- Real-time sentiment analysis from social media platforms
- Alternative data sources (satellite data, credit card spending, etc.)
- Automated financial model building and scenario analysis
- Industry-specific databases and metrics
- Options flow and derivatives analysis tools
- ESG and sustainability metrics analysis
- Cryptocurrency and digital asset analysis
- Or any other specialized analytical capability you deem essential

## Output Format:
You must and can only return a JSON object in the following format enclosed by <Output> and </Output> like:
<Output>
{{
    "tool_name": string, # tool name
    "properties": dict, # tool execution arguments
}}
</Output>

## Tool Usage Rules:
You must always follow these rules to complete the task:
1. After receiving a user task, you will first create an action plan, then call tools according to the action plan to complete the task.
2. Always provide tool calls, otherwise it will fail.
3. Always use correct tool parameters. Do not use variable names in action parameters, use specific values instead.
    - When using the corp_info tool, pay attention to whether its stock_code parameter is valid. If invalid, you need to convert it to a valid format.
4. Never repeat calls to tools that have already been used with exactly the same parameters
5. Do not return any other text format, do not explain your choices, do not apologize, do not express inability to answer.
6. If a step requires multiple tools, choose the most important one.
7. If you have completed all action plans and obtained sufficient information, please use the tool action named "final_report" to provide the final report to the task. This is the only way to complete the task, otherwise you will fall into a loop.
8. If you need to output string, please output in language: {output_language}

Note: 
- Only propose new tools if you identify critical gaps that cannot be addressed by current available tools.
- Be specific about the capabilities and analytical value of any proposed tools.
- Focus on tools that would significantly enhance your research quality and depth.
"""

prompt_for_research_write_result = """
你是一个专业的股票研究分析师。基于你的研究计划和工具调用结果，写出最终的研究报告。

# 研究任务
{task}

# 背景信息
{background_information}

# 研究计划
{plan}

# 工具调用上下文
{tool_call_context}

{hallucination_warning}

# 严格的数据引用和幻觉防护要求

## 数据引用格式标准
你必须严格按照以下格式引用所有数据和信息：
- **工具数据引用**: [工具名称|数据时间|具体数值] 例如：[price_info|2024-01-15 10:30|股价$150.25]
- **新闻信息引用**: [新闻源|发布时间|标题] 例如：[Yahoo Finance|2024-01-15|Apple Reports Strong Q4 Earnings]
- **财务数据引用**: [数据源|报告期|指标名称|数值] 例如：[SEC Filing|Q3 2024|Revenue|$89.5B]
- **市场数据引用**: [数据源|时间戳|指标|数值] 例如：[Market Data|2024-01-15 15:59|Volume|2.5M shares]

## 不确定性量化要求
对于每个关键判断，你必须明确标注不确定性级别：
- **高确定性** (90-100%): 基于多个可靠数据源的一致证据
- **中等确定性** (70-89%): 基于部分数据支持，但存在一些不确定因素
- **低确定性** (50-69%): 基于有限数据或存在相互矛盾的信息
- **高度不确定** (<50%): 缺乏足够数据支持或数据质量存疑

## 严禁行为
1. **绝对禁止编造数据**: 不得创造任何未在工具调用结果中出现的具体数值、日期、公司名称或事件
2. **禁止模糊引用**: 不得使用"据报道"、"市场传言"、"分析师认为"等无法验证的表述
3. **禁止推测性断言**: 不得将推测表述为确定事实
4. **禁止过度外推**: 不得基于有限数据做出超出合理范围的结论

## 数据验证检查清单
在生成每个证据点时，请自我检查：
- [ ] 该数据是否直接来自工具调用结果？
- [ ] 是否提供了完整的数据引用格式？
- [ ] 是否标注了适当的不确定性级别？
- [ ] 是否区分了事实和推论？

# 输出要求
请严格按照以下JSON格式输出你的研究结果：

```json
{{
    "has_opportunity": true/false,
    "action": "buy/sell/hold",
    "symbol_code": "股票代码",
    "evidence_list": [
        "[数据引用格式] 具体证据描述 [确定性级别: XX%]",
        "[数据引用格式] 具体证据描述 [确定性级别: XX%]",
        "[数据引用格式] 具体证据描述 [确定性级别: XX%]"
    ],
    "limitations": "分析的局限性和风险提示，包括数据质量评估和时效性说明",
    "probability": 0.0-1.0,
    "data_quality_assessment": {
        "overall_confidence": "high/medium/low",
        "data_sources_count": "使用的数据源数量",
        "data_freshness": "数据时效性评估",
        "potential_biases": "潜在的数据偏差或局限性"
    }
}}
```

请确保：
1. 每个证据都有完整的数据引用和不确定性标注
2. 诚实评估数据质量和分析局限性
3. 概率评估要与证据强度和不确定性水平一致
4. 在limitations中明确说明哪些关键信息缺失或不确定
"""

format_for_symbol_retrieval = """
<stock>
<market>xxx</market>   # market name, e.g. "CN-Stock", "CN-ETF", "HK-Stock", "US-Stock"
<code>xxx</code>
<name>xxx</name>
<reason>xxx</reason>
</stock>
<stock>
<market>xxx</market>
<code>xxx</code>
<name>xxx</name>
<reason>xxx</reason>
</stock>
...
"""

prompt_for_data_analysis_summary_doc = """
Current time is: {trigger_datetime}

Please perform {summary_style} on the following financial documents, extracting key factual information:

{doc_context}

Requirements:
1. {bias_instruction}
2. Extract specific facts, data, and key information
3. While maintaining accuracy, prioritize content related to the goal
4. Organize content by information importance and timeliness
5. Control within {summary_target_tokens} words
6. For each factual description, add corresponding reference tags at the end, such as [1][2]
7. Output result in language: {language}

{summary_style}:
"""

prompt_for_data_analysis_filter_doc = """
Current time is: {trigger_datetime}

Please select the {titles_to_select} most informative documents from the following financial document titles:

{titles_context}

Selection criteria:
1. Contains specific factual information and data
2. Involves important policies, company dynamics, industry changes
3. Information timeliness and importance
4. Avoid repetitive and low-quality content
5. Output result in language: {language}

Please directly output the selected document IDs, separated by commas, such as: 1,5,8,12
"""

prompt_for_data_analysis_merge_summary = """
Current time is: {trigger_time}
Analysis Goal: {goal_instruction}

Please merge the following multiple document batch summaries into a unified market information factor:

{combined_summary}

Requirements:
1. Merge duplicate information, retain all important facts
2. Sort by information importance and timeliness
3. {summary_focus}
4. Control within {final_target_tokens} words
5. Form clear market information summary
6. Preserve reference identifiers [numbers] format from original text
7. Output result in language: {language}

Please output {final_description} directly, do not include any other content.
{final_description}:
"""

prompt_for_research_invest_task = """
As a professional researcher with specific belief, you need to find opportunities in the market today. You need to submit up to 5 critical analysis suggestions to the investor.

Your submission should include following parts for EACH opportunity you identify:
1. Does valuable opportunity exist in the market today?
2. Symbol Information of the opportunity
3. Evidence list you find to prove the opportunity is valuable. Judger will use these evidences to judge the opportunity is valuable or not.
4. Based on the evidence_list, you need to give a probability to this opportunity.
5. You need to give a limitation to your suggestion, such as risk, etc. No limitation will be rejected.
6. You should provide between 1 to 5 opportunity suggestions based on what you find in the market. Only submit signals for opportunities you genuinely identify.
7. If accepted, your suggestions will execute when the market opens and hold for one day. So you need to focus on short-term information.
8. Each signal should be independent and focus on different stocks or strategies.
9. If you cannot find 5 valuable opportunities, submit fewer high-quality signals rather than padding with low-quality ones.
"""

prompt_for_research_invest_output_format = """
<signals>
<signal>
<has_opportunity>xxx</has_opportunity>  # yes or no
<action>xxx</action>  # buy or sell
<symbol_code>xxx</symbol_code>     # such as 600519.SH or TSLA
<symbol_name>xxx</symbol_name>  # such as Apple Inc or Tesla
<evidence_list>        # no more than 20 evidences
<evidence>xxx</evidence>   # a detailed evidence description, including convincing logical inferences which support your suggestion. About 100 words.
<time>xxx</time>           # evidence time
<from_source>xxx</from_source>   # evidence source, from which media name or website name or tools name
...
</evidence_list>
<limitations>
<limitation>xxx</limitation>   # limitations of your suggestion, such as risk, etc.
...
</limitations>
<probability>xxx</probability>  # 0-100
</signal>
<!-- Repeat <signal>...</signal> block for each opportunity you identify, up to 5 signals -->
<!-- Only include signals for genuine opportunities you find in the market -->
</signals>
"""