import tiktoken
encoding = tiktoken.get_encoding("cl100k_base")


def count_tokens(text):
    """
    计算文本的token数量
    
    Args:
        text (str): 要计算token的文本
        
    Returns:
        int: token数量
    """
    if not text or not isinstance(text, str):
        return 0
    try:
        return len(encoding.encode(text))
    except Exception as e:
        print(f"Token计算错误: {e}")
        return 0
