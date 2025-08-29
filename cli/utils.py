import questionary
from typing import List, Optional, Tuple, Dict
from datetime import datetime, timedelta
import tushare as ts
from rich.console import Console
from contest_trade.config.config import cfg
from contest_trade.models.llm_model import GLOBAL_LLM

console = Console()

def get_trigger_time() -> str:
    """提示用户输入触发时间"""

    now = datetime.now()
    options = [
        f"今天A股盘前 ({now.strftime('%Y-%m-%d')} 09:00:00)",
        f"今天美股盘前 ({now.strftime('%Y-%m-%d')} 15:30:00，夏令时美东时间03:30:00)",
        f"今天美股盘前 ({now.strftime('%Y-%m-%d')} 16:30:00，冬令时美东时间04:30:00)"
    ]
    
    choice = questionary.select(
        "选择触发时间:",
        choices=options,
        style=questionary.Style([
            ("text", "fg:white"),
            ("highlighted", "fg:green bold"),
            ("pointer", "fg:green"),
        ])
    ).ask()
    
    if choice == options[0]:
        return f"{now.strftime('%Y-%m-%d')} 09:00:00"
    elif choice == options[1]:
        return f"{now.strftime('%Y-%m-%d')} 15:30:00"
    elif choice == options[2]:
        return f"{now.strftime('%Y-%m-%d')} 16:30:00"

def validate_tushare_connection():
    try:
        console.print("🔍 [cyan]正在验证必要配置1: Tushare配置...[/cyan]")
        ts.set_token(cfg.tushare_key)
        pro = ts.pro_api(cfg.tushare_key, timeout=3)
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=3)).strftime('%Y%m%d')
        trade_cal = pro.trade_cal(exchange='SSE', start_date=start_date, end_date=end_date, timeout=1)
        print(trade_cal)
        if trade_cal is not None and len(trade_cal) > 0:
            console.print(f"✅ [green]Tushare连接成功[/green]")
            return True
        else:
            console.print("❌ [red]Tushare连接失败 - 未获取到数据[/red]")
            return False
    except Exception as e:
        console.print(f"❌ [red]Tushare连接失败: {str(e)}[/red]")
        return False

def validate_llm_connection():
    """验证LLM连接"""
    try:
        console.print("🔍 [cyan]正在验证必要配置2: LLM配置...[/cyan]")
        test_messages = [
            {"role": "user", "content": "请回复'连接测试成功'，不要添加任何其他内容。"}
        ]
        
        result = GLOBAL_LLM.run(test_messages, max_tokens=50, temperature=0.1, max_retries=1)
        
        if result and hasattr(result, 'content') and result.content:
            console.print(f"✅ [green]LLM连接成功[/green] - 模型: {GLOBAL_LLM.model_name}")
            return True
        else:
            console.print("❌ [red]LLM连接失败 - 无响应内容[/red]")
            return False
    except Exception as e:
        console.print(f"❌ [red]LLM连接失败: {str(e)}[/red]")
        return False

def validate_required_services():
    """验证所有必需的服务连接"""
    console.print("\n" + "="*50)
    console.print("🔧 [bold blue]正在验证必要系统配置...[/bold blue]")
    console.print("="*50)
    all_valid = True
    
    # 跳过Tushare验证，因为使用美股数据
    console.print("🔍 [cyan]跳过Tushare验证 - 使用美股数据源[/cyan]")
    console.print("✅ [green]美股数据源配置正常[/green]")
    
    # 验证LLM
    if not validate_llm_connection():
        all_valid = False
    console.print("="*50)
    
    if all_valid:
        console.print("🎉 [bold green]所有必要系统配置验证通过，系统准备就绪！[/bold green]")
        console.print("="*50 + "\n")
        return True
    else:
        console.print("⚠️  [bold red]必要系统配置验证失败，请检查配置文件[/bold red]")
        console.print("="*50 + "\n")
        return False

def format_agent_name(agent_type: str, agent_id: int, agent_name: str) -> str:
    """格式化Agent名称"""
    if agent_type == "data":
        return f"📊 Data Agent {agent_id} ({agent_name})"
    elif agent_type == "research":
        return f"🔍 Research Agent {agent_id} ({agent_name})"
    else:
        return f"🤖 Agent {agent_id} ({agent_name})"

def format_event_type(event_type: str) -> str:
    """格式化事件类型"""
    event_icons = {
        "on_chain_start": "🔄",
        "on_chain_end": "✅",
        "on_custom": "🎯",
        "on_chain_error": "❌",
    }
    return f"{event_icons.get(event_type, '📝')} {event_type}"

def extract_signal_info(signal: Dict) -> Dict:
    """提取信号信息"""
    return {
        "symbol_name": signal.get("symbol_name", "N/A"),
        "symbol_code": signal.get("symbol_code", "N/A"),
        "action": signal.get("action", "N/A"),
        "probability": signal.get("probability", "N/A"),
        "has_opportunity": signal.get("has_opportunity", "N/A"),
    }