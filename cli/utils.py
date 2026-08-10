import questionary
from typing import List, Optional, Tuple, Dict
from datetime import datetime, timedelta
import tushare as ts
from rich.console import Console
import os

console = Console()

def get_trigger_time() -> str:
    """提示用户输入触发时间"""
    now = datetime.now()
    time_options = [
        f"A股当前时间 ({now.strftime('%Y-%m-%d %H:%M:%S')})",
        # f"今天美股盘前 ({now.strftime('%Y-%m-%d')} 15:30:00，夏令时美东时间03:30:00)",
        # f"今天美股盘前 ({now.strftime('%Y-%m-%d')} 16:30:00，冬令时美东时间04:30:00)"
    ]
    
    time_choice = questionary.select(
        "选择触发时间:（其他时间请期待后续版本）",
        choices=time_options,
        style=questionary.Style([
            ("text", "fg:white"),
            ("highlighted", "fg:green bold"),
            ("pointer", "fg:green"),
        ])
    ).ask()
    
    if time_choice == time_options[0]:
        return f"{now.strftime('%Y-%m-%d %H:%M:%S')}"
    else:
        return f"{now.strftime('%Y-%m-%d %H:%M:%S')}"

def validate_tushare_connection():
    """验证Tushare连接"""
    try:
        # Import config when needed
        from contest_trade.config.config import cfg
        
        console.print("🔍 [cyan]正在验证Tushare配置...[/cyan]")
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
        from contest_trade.models.llm_model import GLOBAL_LLM, GLOBAL_LLM_CONFIG
        from contest_trade.models.provider_auth import (
            format_missing_key_error,
            get_env_config_hints,
            get_env_var_candidates,
            resolve_api_key,
        )

        console.print("🔍 [cyan]正在验证LLM配置...[/cyan]")
        console.print(
            f"  Provider: {GLOBAL_LLM_CONFIG.provider} | "
            f"Model: Kimi K3 ({GLOBAL_LLM_CONFIG.model_name}) | "
            f"Base URL: {GLOBAL_LLM_CONFIG.base_url or '—'}"
        )

        resolved_key = resolve_api_key(
            GLOBAL_LLM_CONFIG.provider,
            config_key=GLOBAL_LLM_CONFIG.api_key,
            base_url=GLOBAL_LLM_CONFIG.base_url,
        )
        if not resolved_key:
            env_vars = get_env_var_candidates(
                GLOBAL_LLM_CONFIG.provider,
                GLOBAL_LLM_CONFIG.base_url,
            )
            console.print(
                f"❌ [red]{format_missing_key_error(GLOBAL_LLM_CONFIG.provider, GLOBAL_LLM_CONFIG.base_url)}[/red]"
            )
            console.print(f"   请在 .env 中设置: {', '.join(env_vars[:3])}")
            for hint in get_env_config_hints(
                GLOBAL_LLM_CONFIG.provider,
                GLOBAL_LLM_CONFIG.base_url,
            ):
                console.print(f"   ⚠️  [yellow]{hint}[/yellow]")
            return False

        test_messages = [
            {"role": "user", "content": "请回复'连接测试成功'，不要添加任何其他内容。"}
        ]
        result = GLOBAL_LLM.run(test_messages, max_tokens=32, temperature=1, max_retries=0)
        has_content = result and (
            (hasattr(result, 'content') and result.content)
            or (hasattr(result, 'reasoning_content') and result.reasoning_content)
        )
        if has_content:
            console.print(f"✅ [green]LLM连接成功[/green] - 模型: Kimi K3 ({GLOBAL_LLM.model_name})")
            return True
        else:
            console.print("❌ [red]LLM连接失败 - 无响应内容[/red]")
            return False
    except Exception as e:
        console.print(f"❌ [red]LLM连接失败: {str(e)}[/red]")
        return False

def validate_required_services():
    """根据配置文件中的tushare_key自动决定验证策略"""
    # Import config when needed
    from contest_trade.config.config import cfg
    
    console.print("\n" + "="*50)
    console.print("🔧 [bold blue]正在验证必要系统配置...[/bold blue]")
    console.print("="*50)
    all_valid = True
    
    # 检查tushare_key是否为空
    tushare_key = getattr(cfg, 'tushare_key', '')
    has_tushare_key = tushare_key and tushare_key.strip() and tushare_key != "YOUR_TUSHARE_KEY"
    
    if has_tushare_key:
        console.print("🔍 [cyan]检测到Tushare密钥，将验证Tushare连接...[/cyan]")
        # 验证Tushare
        if not validate_tushare_connection():
            all_valid = False
    else:
        console.print("ℹ️  [yellow]未检测到Tushare密钥，跳过Tushare验证[/yellow]")

        # 验证akshare是否安装
        try:
            import akshare
            console.print("✅ [green]akshare已安装[/green]")
        except ImportError:
            console.print("❌ [red]akshare未安装，请重新执行pip install akshare[/red]")
            all_valid = False
    
    # 始终验证LLM
    if not validate_llm_connection():
        all_valid = False
    
    console.print("="*50)
    
    if all_valid:
        if has_tushare_key:
            console.print("🎉 [bold green]所有必要系统配置验证通过（包含Tushare），系统准备就绪！[/bold green]")
        else:
            console.print("🎉 [bold green]所有必要系统配置验证通过（不含Tushare），系统准备就绪！[/bold green]")
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

def get_market_selection() -> str:
    """获取用户市场选择 - 使用箭头键选择"""
    market_options = [
        "CN-Stock (A股市场)",
        "US-Stock (美股市场)"
    ]
    
    market_choice = questionary.select(
        "请选择要分析的市场(Please select a market to analyze):",
        choices=market_options,
        style=questionary.Style([
            ("text", "fg:white"),
            ("highlighted", "fg:green bold"),
            ("pointer", "fg:green"),
        ])
    ).ask()
    
    # 如果用户取消选择
    if market_choice is None:
        return None
    
    # 根据选择返回对应的市场代码
    if market_choice == market_options[0]:
        return "CN-Stock"
    elif market_choice == market_options[1]:
        return "US-Stock"
    else:
        return None

def get_trigger_time_for_market(market: str) -> str:
    """根据市场获取对应的触发时间，并设置环境变量"""
    # 设置环境变量
    os.environ['CONTEST_TRADE_MARKET'] = market
    
    # 根据市场获取触发时间
    if market == "CN-Stock":
        # A股市场使用当前交易日
        return get_trigger_time()
    elif market == "US-Stock":
        # 美股市场使用美东时区时间
        from datetime import datetime, timezone, timedelta
        
        try:
            # 尝试使用 pytz 获取美东时区
            import pytz
            eastern_tz = pytz.timezone('America/New_York')
            now = datetime.now(eastern_tz)
            console.print(f"🇺🇸 [cyan]使用美东时区: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}[/cyan]")
        except ImportError:
            # 如果没有 pytz，使用简单的时区计算（考虑夏令时）
            from datetime import datetime
            import time
            
            # 检查是否为夏令时（简化版本：3月第二个周日到11月第一个周日）
            now_utc = datetime.now(timezone.utc)
            is_dst = time.daylight and time.localtime().tm_isdst > 0
            
            if is_dst:
                # 夏令时 EDT = UTC-4
                offset_hours = -4
                tz_name = "EDT"
            else:
                # 标准时间 EST = UTC-5  
                offset_hours = -5
                tz_name = "EST"
            
            eastern_tz = timezone(timedelta(hours=offset_hours))
            now = now_utc.astimezone(eastern_tz)
            console.print(f"🇺🇸 [cyan]使用美东时区: {now.strftime('%Y-%m-%d %H:%M:%S')} {tz_name}[/cyan]")
        
        return now.strftime("%Y-%m-%d %H:%M:%S")
    else:
        return None

