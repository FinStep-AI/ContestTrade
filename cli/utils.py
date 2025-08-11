import questionary
from typing import List, Optional, Tuple, Dict
from datetime import datetime
import re
from rich.console import Console

from .models import AgentType, AgentStatus

console = Console()


def get_trigger_time() -> str:
    """提示用户输入触发时间"""
    def validate_datetime(datetime_str: str) -> bool:
        try:
            datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
            return True
        except ValueError:
            return False

    # 提供预设选项
    now = datetime.now()
    options = [
        f"现在 ({now.strftime('%Y-%m-%d %H:%M:%S')})",
        f"今天盘前 ({now.strftime('%Y-%m-%d')} 09:00:00)",
        f"昨天收盘 ({now.strftime('%Y-%m-%d')} 15:00:00)",
        "自定义时间",
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
    
    if choice == options[0]:  # 现在
        return now.strftime('%Y-%m-%d %H:%M:%S')
    elif choice == options[1]:  # 今天盘前
        return f"{now.strftime('%Y-%m-%d')} 09:00:00"
    elif choice == options[2]:  # 昨天收盘
        return f"{now.strftime('%Y-%m-%d')} 15:00:00"
    else:  # 自定义时间
        trigger_time = questionary.text(
            "请输入自定义触发时间 (YYYY-MM-DD HH:MM:SS):",
            default=now.strftime('%Y-%m-%d %H:%M:%S'),
            validate=lambda x: validate_datetime(x.strip()) or "请输入有效的时间格式 YYYY-MM-DD HH:MM:SS",
            style=questionary.Style([
                ("text", "fg:green"),
                ("highlighted", "noinherit"),
            ])
        ).ask()

        if not trigger_time:
            console.print("\n[red]未提供触发时间，退出...[/red]")
            exit(1)

        return trigger_time.strip()


def validate_config() -> bool:
    """验证配置"""
    try:
        import sys
        import os
        # 添加项目根目录到Python路径
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        from contest_trade.config.config import cfg
        return True
    except ImportError as e:
        console.print(f"[red]配置加载失败: {e}[/red]")
        return False
    except Exception as e:
        console.print(f"[red]配置验证失败: {e}[/red]")
        return False


def format_agent_name(agent_type: str, agent_id: int, agent_name: str) -> str:
    """格式化代理名称"""
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