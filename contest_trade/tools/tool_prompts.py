FINANCIAL_TOOL_SELECT_PROMPT = """You are an expert in breaking down tool calls, specializing in decomposing company financial information query tasks into tool call parameters.

Today is {date} 

## Task:
{task}

## Market:
{market}

## Stock Symbol:
{symbol}


## Task Execution
To complete the task, you need to write tool calls targeting the user task instruction as your goal.
The tool call you write is an action: after tool execution, you will receive the tool call results as "Observation".
Here's an example of using tools:
- valid。
---
User task: "Guangzhou population data"
---
<Output>
{{
    "name": "search",
    "arguments": {{"query": "Guangzhou population"}}
}}
</Output>

The above example uses a conceptual tool that you may not have. You can only use the following tools:
{tool_info}

Please select the most appropriate tool based on the task content and stock code, and provide all the parameters required for that tool call.

Output in JSON format enclosed by <Output> and </Output> like this:
<Output>
{{
    "tool_name": string, # tool name
    "properties": dict, # tool execution arguments
}}
</Output>

- notice: only one tool is allowed to be called.
"""

STOCK_FILTER_PROMPT = """
<stock_dataframe_schema>
字段名              类型      含义                example
ts_code            str      股票代码              000001.SZ
name               str      股票名称              平安银行
area               str      地区                  深圳
industry           str      行业                  银行
turnover_rate      float    换手率（%）           2.06
turnover_rate_f    float    换手率（自由流通股）   3.28
volume_ratio       float    量比                  1.08
pe                 float    市盈率（总市值/净利润，亏损的PE为空） 21.22
pe_ttm             float    市盈率（TTM，亏损的PE为空） 22.06
pb                 float    市净率（总市值/净资产） 2.16
ps                 float    市销率                2.4
ps_ttm             float    市销率（TTM）         2.37
dv_ratio           float    股息率（%）           1.45
dv_ttm             float    股息率（TTM）（%）    1.11
total_share        float    总股本（万股）        34813.69
float_share        float    流通股本（万股）      32001.01
free_share         float    自由流通股本（万）    20114.79
total_mv           float    总市值（万元）        470332.97
circ_mv            float    流通市值（万元）      432333.64
</stock_dataframe_schema>

<task>
根据用户的query，生成一个函数用于筛选股票。这个函数会被python执行器执行。因此只需要生成函数即可。
- 生成的代码满足output_template的模板。严格按照模板生成，不要有任何其他内容。
- 最好是按照某个数值字段进行排序，系统会默认选取排序后的前面30只股票。
- 只输出ts_code,name和进行过筛选的字段。其他字段不需要显示。
</task>

<output_template>
```python
def filter_stock(stock_dataframe: pd.DataFrame) -> pd.DataFrame:
    # your filter code here
    return filtered_df
```
</output_template>

<query>
{query}
</query>
"""