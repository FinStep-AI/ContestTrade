"""
字符串工具
"""

def convert_to_tushare_format(stock_code: str) -> str:
    """
    将600519.SH格式转换为tushare格式(sh600519)
    """
    if not stock_code or '.' not in stock_code:
        raise ValueError(f"股票代码格式错误: {stock_code}，应为600519.SH格式")
    
    code, market = stock_code.split('.')
    
    market_mapping = {
        'SH': 'sh',
        'SZ': 'sz', 
        'BJ': 'bj'
    }
    
    if market not in market_mapping:
        raise ValueError(f"不支持的市场代码: {market}，应为SH/SZ/BJ")
    
    return f"{market_mapping[market]}{code}" 
