"""
ContestTrade CLI: 基于内部竞赛机制的多代理交易系统
"""
import asyncio
import sys
import os
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime
from collections import deque

import typer
from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout
from rich.text import Text
from rich.align import Align
from rich import box

from .models import AgentType, AgentStatus, ContestResult
from .utils import get_trigger_time, validate_config
from contest_trade.config.config import cfg, PROJECT_ROOT
sys.path.append(str(PROJECT_ROOT))
from contest_trade.run_company_simple import SimpleTradeCompany

# 创建控制台
console = Console()

app = typer.Typer(
    name="contesttrade",
    help="ContestTrade CLI: 基于内部竞赛机制的多代理交易系统",
    add_completion=True,
)


class ContestTradeDisplay:
    """ContestTrade显示管理器"""
    
    def __init__(self):
        self.messages = deque(maxlen=100)  # 增加消息数量以显示更多步骤
        # 预设代理状态
        self.agent_status = {
            # 数据代理
            "thx_summary_agent": "pending",
            "sina_summary_agent": "pending",
            "price_market_agent": "pending", 
            "hot_money_agent": "pending",
            # 研究代理
            "agent_0": "pending",
            "agent_1": "pending",
            "agent_2": "pending"
        }
        self.agent_sub_status = {
            # 数据代理子状态
            "thx_summary_agent": "等待启动",
            "sina_summary_agent": "等待启动",
            "price_market_agent": "等待启动",
            "hot_money_agent": "等待启动",
            # 研究代理子状态
            "agent_0": "等待启动",
            "agent_1": "等待启动", 
            "agent_2": "等待启动"
        }
        self.current_task = "初始化系统..."
        self.progress_info = ""
        self.final_state = None
        self.analysis_completed = False
        self.step_counts = {"data": 0, "research": 0, "contest": 0, "finalize": 0}
        
    def add_message(self, message_type: str, content: str):
        """添加消息"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.messages.append(f"[{timestamp}] {message_type}: {content}")
        
    def update_agent_status(self, agent_name: str, status: str, sub_status: str = ""):
        """更新代理状态和子状态"""
        # 确保代理在状态字典中
        if agent_name not in self.agent_status:
            self.agent_status[agent_name] = "pending"
            self.agent_sub_status[agent_name] = ""
            
        self.agent_status[agent_name] = status
        self.agent_sub_status[agent_name] = sub_status
        
    def set_current_task(self, task: str):
        """设置当前任务"""
        self.current_task = task
        
    def set_progress_info(self, info: str):
        """设置进度信息"""
        self.progress_info = info
        
    def set_analysis_completed(self, completed: bool = True):
        """设置分析完成状态"""
        self.analysis_completed = completed
        
    def create_layout(self, trigger_time: str) -> Layout:
        """创建自适应布局"""
        layout = Layout()
        
        # 创建多行布局
        layout.split_column(
            Layout(name="header", size=8),
            Layout(name="main_content")
        )
        
        # 主要内容区域分割为左右两列，按照1:2的比例
        layout["main_content"].split_row(
            Layout(name="left_panel", ratio=1),
            Layout(name="right_panel", ratio=2)
        )
        
        # 左侧面板分割
        layout["left_panel"].split_column(
            Layout(name="status", ratio=3),
            Layout(name="progress", ratio=2)
        )
        
        # 右侧面板分割
        layout["right_panel"].split_column(
            Layout(name="content", ratio=3),
            Layout(name="footer", ratio=1)
        )
        
        return layout
        
    def update_display(self, layout: Layout, trigger_time: str):
        """更新显示"""
        # 读取欢迎信息
        welcome_text = Path(__file__).parent / "static" / "welcome.txt"
        if welcome_text.exists():
            with open(welcome_text, "r", encoding="utf-8") as f:
                welcome = f.read()
        else:
            welcome = "ContestTrade: 基于内部竞赛机制的多代理交易系统"
        
        # 更新顶部标题 - 自适应宽度，与下方布局对齐
        header_panel = Panel(
            Align.center(welcome),
            title="🎯 ContestTrade - 基于内部竞赛机制的多代理交易系统",
            border_style="blue",
            padding=(0, 1),
            expand=True  # 自适应宽度
        )
        layout["header"].update(header_panel)
        
        # 更新代理状态面板
        status_text = Text()
        
        # 数据代理状态
        data_agents = {k: v for k, v in self.agent_status.items() if not k.startswith("agent_")}
        if data_agents:
            status_text.append("📊 数据代理\n", style="bold cyan")
            for agent_name, status in data_agents.items():
                status_icon = {
                    "pending": "⏳",
                    "running": "🔄", 
                    "completed": "✅",
                    "failed": "❌"
                }.get(status, "❓")
                
                agent_display = agent_name[:20].ljust(20)
                status_text.append(f"{agent_display} {status_icon}\n", style="dim")
                
                # 显示子状态
                sub_status = self.agent_sub_status.get(agent_name, "")
                if sub_status:
                    status_text.append(f"{'  ':<22}└─ {sub_status}\n", style="dim blue")
        
        # 研究代理状态
        research_agents = {k: v for k, v in self.agent_status.items() if k.startswith("agent_")}
        if research_agents:
            status_text.append("\n🔍 研究代理\n", style="bold green")
            for agent_name, status in research_agents.items():
                status_icon = {
                    "pending": "⏳",
                    "running": "🔄", 
                    "completed": "✅",
                    "failed": "❌"
                }.get(status, "❓")
                
                agent_display = agent_name[:20].ljust(20)
                status_text.append(f"{agent_display} {status_icon}\n", style="dim")
                
                # 显示子状态
                sub_status = self.agent_sub_status.get(agent_name, "")
                if sub_status:
                    status_text.append(f"{'  ':<22}└─ {sub_status}\n", style="dim green")
        
        status_panel = Panel(
            status_text,
            title="🤖 代理状态",
            border_style="yellow",
            padding=(0, 1),
            expand=True  # 自适应宽度
        )
        layout["status"].update(status_panel)
        
        # 更新进度面板
        progress_text = Text()
        progress_text.append(f"📅 触发时间: {trigger_time}\n", style="cyan")
        progress_text.append(f"🎯 当前任务: {self.current_task}\n", style="yellow")
        if self.progress_info:
            progress_text.append(f"📈 进度: {self.progress_info}\n", style="green")
        
        # 显示步骤计数
        progress_text.append(f"\n📊 步骤统计:\n", style="bold blue")
        progress_text.append(f"  数据代理事件: {self.step_counts['data']}\n", style="dim")
        progress_text.append(f"  研究代理事件: {self.step_counts['research']}\n", style="dim")
        progress_text.append(f"  竞赛事件: {self.step_counts['contest']}\n", style="dim")
        progress_text.append(f"  完成事件: {self.step_counts['finalize']}\n", style="dim")
        
        progress_panel = Panel(
            progress_text,
            title="📊 进度信息",
            border_style="blue",
            padding=(0, 1),
            expand=True  # 自适应宽度
        )
        layout["progress"].update(progress_panel)
        
        # 更新主内容区域
        content_text = Text()
        content_text.append("🔄 实时事件日志\n", style="bold blue")
        
        if self.messages:
            for msg in list(self.messages)[-15:]:  # 显示最近15条消息
                content_text.append(f"{msg}\n", style="dim")
        else:
            content_text.append("  ⏳ 等待事件...\n", style="dim")
        
        content_panel = Panel(
            content_text,
            title="📄 事件流",
            border_style="blue",
            padding=(1, 2),
            expand=True  # 自适应宽度
        )
        layout["content"].update(content_panel)
        
        # 更新底部
        if self.analysis_completed and self.final_state:
            footer_text = self._create_result_summary()
            footer_title = "🏆 结果摘要"
        else:
            footer_text = Text()
            footer_text.append("🔄 分析进行中...", style="bold yellow")
            if self.analysis_completed:
                footer_text.append("\n✅ 分析完成！按回车键退出运行界面...", style="bold green")
            footer_title = "📊 状态信息"
        
        footer_panel = Panel(
            footer_text,
            title=footer_title,
            border_style="green",
            padding=(0, 1),
            expand=True  # 自适应宽度
        )
        layout["footer"].update(footer_panel)
    
    def _create_result_summary(self) -> Text:
        """创建结果摘要"""
        summary_text = Text()
        
        if self.final_state:
            # 从step_results中获取统计信息
            step_results = self.final_state.get('step_results', {})
            data_team_results = step_results.get('data_team', {})
            research_team_results = step_results.get('research_team', {})
            
            data_factors_count = data_team_results.get('factors_count', 0)
            research_signals_count = research_team_results.get('signals_count', 0)
            total_events_count = data_team_results.get('events_count', 0) + research_team_results.get('events_count', 0)
            
            summary_text.append(f"📊 数据因子: {data_factors_count} | ", style="green")
            summary_text.append(f"🔍 研究信号: {research_signals_count} | ", style="blue")
            summary_text.append(f"📈 总事件: {total_events_count}\n", style="yellow")
            
            # 显示最佳信号
            best_signals = step_results.get('contest', {}).get('best_signals', [])
            if best_signals:
                summary_text.append("🎯 最佳信号: ", style="bold red")
                for i, signal in enumerate(best_signals[:3]):
                    symbol_name = signal.get('symbol_name', 'N/A')
                    action = signal.get('action', 'N/A')
                    probability = signal.get('probability', 'N/A')
                    summary_text.append(f"{symbol_name}({action}-{probability}%)", style="cyan")
                    if i < min(2, len(best_signals) - 1):
                        summary_text.append(", ", style="dim")

            summary_text.append("\n💡分析完成，按回车退出运行界面...")
        else:
            summary_text.append("❌ 分析失败", style="red")
        
        return summary_text


def run_contest_analysis_interactive(trigger_time: str):
    """在交互界面中运行竞赛分析"""
    try:
        # 创建显示管理器
        display = ContestTradeDisplay()
        
        # 创建布局
        layout = display.create_layout(trigger_time)
        
        # 使用Live界面运行
        with Live(layout, refresh_per_second=3, screen=True, auto_refresh=True, console=console) as live:
            # 初始显示
            display.update_display(layout, trigger_time)
            
            # 添加初始消息
            display.add_message("系统", f"开始分析时间: {trigger_time}")
            display.set_current_task("初始化ContestTrade系统...")
            display.set_progress_info("系统启动中...")
            display.update_display(layout, trigger_time)
            
            # 检查模块导入
            try:
                if SimpleTradeCompany is None:
                    raise ImportError("SimpleTradeCompany模块导入失败")
                    
                display.add_message("系统", "✅ 成功导入SimpleTradeCompany模块")
                display.update_display(layout, trigger_time)
                
                # 创建公司实例
                company = SimpleTradeCompany()
                display.add_message("系统", "✅ 成功创建SimpleTradeCompany实例")
                display.update_display(layout, trigger_time)
                
            except Exception as e:
                display.add_message("错误", f"❌ 模块导入失败: {str(e)}")
                display.update_display(layout, trigger_time)
                return None, display
            
            # 运行工作流并捕获输出
            final_state = asyncio.run(run_with_events_capture(company, trigger_time, display, layout))
            
            # 运行结束后
            if final_state:
                display.add_message("完成", "✅ 分析完成！")
                display.set_current_task("分析完成，等待用户选择...")
                display.set_analysis_completed(True)
                display.final_state = final_state
                display.update_display(layout, trigger_time)
                
                # 等待用户手动退出
                console.print("\n[green]✅ 分析完成！[/green]")
                console.print("[dim]按任意键退出运行界面...[/dim]")
                input()
                
            else:
                display.add_message("错误", "❌ 分析失败")
                display.set_current_task("分析失败")
                display.update_display(layout, trigger_time)
                console.print("\n[red]❌ 分析失败！[/red]")
                console.print("[dim]按任意键退出运行界面...[/dim]")
                input()
                return None, display
                
    except Exception as e:
        console.print(f"[red]运行失败: {e}[/red]")
        return None, None
    
    # Live界面结束后，处理用户输入
    if final_state:
        return ask_user_for_next_action(final_state)
    
    return final_state, display


async def run_with_events_capture(company, trigger_time: str, display: ContestTradeDisplay, layout):
    """运行公司工作流并捕获事件流"""
    try:
        display.add_message("开始", "🚀 开始运行工作流...")
        display.set_current_task("🔄 启动工作流...")
        display.update_display(layout, trigger_time)
        
        # 运行公司工作流并处理事件
        final_state = None
        async for event in company.run_company_with_events(trigger_time):
            event_name = event.get("name", "")
            event_type = event.get("event", "")
            event_data = event.get("data", {})
            
            # 处理公司级别事件
            if event_name in ["run_data_agents", "run_research_agents", "run_contest", "finalize"]:
                if event_type == "on_chain_start":
                    display.set_current_task(f"🔄 开始 {event_name}")
                    if event_name == "run_data_agents":
                        display.set_progress_info("数据收集阶段 1/4")
                        # 标记数据代理开始运行
                        data_agent_names = ["thx_summary_agent", "sina_summary_agent", "price_market_agent", "hot_money_agent"]
                        for agent_name in data_agent_names:
                            display.update_agent_status(agent_name, "running", "🚀 准备启动")
                    elif event_name == "run_research_agents":
                        display.set_progress_info("研究分析阶段 2/4")
                        # 完成数据代理，开始研究代理
                        data_agent_names = ["thx_summary_agent", "sina_summary_agent", "price_market_agent", "hot_money_agent"]
                        for agent_name in data_agent_names:
                            display.update_agent_status(agent_name, "completed", "✅ 完成")
                        research_agent_names = ["agent_0", "agent_1", "agent_2"]
                        for agent_name in research_agent_names:
                            display.update_agent_status(agent_name, "running", "🚀 准备启动")
                    elif event_name == "run_contest":
                        display.set_progress_info("竞赛评选阶段 3/4")
                        # 完成研究代理
                        research_agent_names = ["agent_0", "agent_1", "agent_2"]
                        for agent_name in research_agent_names:
                            display.update_agent_status(agent_name, "completed", "✅ 完成")
                    elif event_name == "finalize":
                        display.set_progress_info("结果生成阶段 4/4")
                        
                elif event_type == "on_chain_end":
                    display.set_current_task(f"✅ 完成 {event_name}")
                    if event_name == "finalize":
                        final_state = event_data.get("output", {})
                        display.set_analysis_completed(True)
                        
            # 处理LangGraph子图事件（代理事件）
            elif event_name == "LangGraph":
                if event_type == "on_chain_start":
                    # 检查是否是代理相关的事件
                    tags = event.get("tags", [])
                    if any("agent" in str(tag).lower() for tag in tags):
                        display.add_message("代理", f"🔄 启动代理子图")
                        if "data" in str(tags).lower():
                            display.step_counts["data"] += 1
                        elif "research" in str(tags).lower():
                            display.step_counts["research"] += 1
                            
                elif event_type == "on_chain_end":
                    tags = event.get("tags", [])
                    if any("agent" in str(tag).lower() for tag in tags):
                        display.add_message("代理", f"✅ 完成代理子图")
            
            # 处理具体的节点事件（显示真实的🔄✅状态）
            elif event_type in ["on_chain_start", "on_chain_end"]:
                # 过滤掉不需要显示的事件
                if event_name not in ["__start__", "__end__"]:
                    emoji = "🔄" if event_type == "on_chain_start" else "✅"
                    
                    # 识别代理类型
                    if any(keyword in event_name.lower() for keyword in ["init_factor", "recompute_factor", "submit_result"]):
                        # 数据代理相关事件
                        agent_type = "数据代理"
                        display.step_counts["data"] += 1
                        
                        # 根据当前数据代理状态更新
                        current_data_agents = [k for k in display.agent_status.keys() if not k.startswith("agent_")]
                        if current_data_agents:
                            for agent_name in current_data_agents:
                                if display.agent_status[agent_name] in ["pending", "running"]:
                                    display.update_agent_status(agent_name, "running", f"{emoji} {event_name}")
                                    break
                                    
                    elif any(keyword in event_name.lower() for keyword in ["init_signal", "recompute_signal"]):
                        # 研究代理相关事件
                        agent_type = "研究代理"
                        display.step_counts["research"] += 1
                        
                        # 根据当前研究代理状态更新
                        current_research_agents = [k for k in display.agent_status.keys() if k.startswith("agent_")]
                        if current_research_agents:
                            for agent_name in current_research_agents:
                                if display.agent_status[agent_name] in ["pending", "running"]:
                                    display.update_agent_status(agent_name, "running", f"{emoji} {event_name}")
                                    break
                    else:
                        agent_type = "系统"
                    
                    display.add_message(agent_type, f"{emoji} {event_name}")
            
            # 处理自定义事件
            elif event_type == "on_custom":
                custom_name = event.get("name", "")
                custom_data = event.get("data", {})
                
                if custom_name.startswith("data_agent_"):
                    agent_id = custom_data.get("agent_id", "unknown")
                    agent_name = custom_data.get("agent_name", "unknown")
                    
                    # 映射真实的agent_name到我们的显示名称
                    display_agent = agent_name  # 默认使用原名
                    if "thx" in agent_name.lower():
                        display_agent = "thx_summary_agent"
                    elif "sina" in agent_name.lower():
                        display_agent = "sina_summary_agent"
                    elif "price" in agent_name.lower():
                        display_agent = "price_market_agent"
                    elif "hot" in agent_name.lower() or "money" in agent_name.lower():
                        display_agent = "hot_money_agent"
                    
                    # 提取事件类型来显示具体任务
                    task_desc = "处理数据"
                    if "on_chain_start" in custom_name:
                        task_desc = "🔄 开始处理"
                    elif "on_chain_end" in custom_name:
                        task_desc = "✅ 完成步骤"
                    elif "_recompute_factor" in custom_name:
                        task_desc = "🔄 重新计算因子"
                    elif "submit_result" in custom_name:
                        task_desc = "🔄 提交结果"
                    
                    display.update_agent_status(display_agent, "running", f"{task_desc}")
                    display.add_message("数据代理", f"📊 {agent_name}: {task_desc}")
                    
                elif custom_name.startswith("research_agent_"):
                    agent_id = custom_data.get("agent_id", "unknown")
                    agent_name = custom_data.get("agent_name", f"agent_{agent_id}")
                    
                    # 提取事件类型来显示具体任务
                    task_desc = "研究分析"
                    if "on_chain_start" in custom_name:
                        task_desc = "🔄 开始研究"
                    elif "on_chain_end" in custom_name:
                        task_desc = "✅ 完成步骤"
                    elif "_recompute_signal" in custom_name:
                        task_desc = "🔄 重新计算信号"
                    elif "submit_result" in custom_name:
                        task_desc = "🔄 提交结果"
                    
                    display.update_agent_status(agent_name, "running", f"{task_desc}")
                    display.add_message("研究代理", f"🔍 {agent_name}: {task_desc}")
            
            # 更新显示
            display.update_display(layout, trigger_time)
        
        # 设置所有代理为完成状态
        for agent_name in display.agent_status:
            display.update_agent_status(agent_name, "completed", "✅ 完成")
        
        return final_state
        
    except Exception as e:
        display.add_message("错误", f"❌ 运行失败: {str(e)}")
        console.print(f"[red]详细错误: {e}[/red]")
        return None


def ask_user_for_next_action(final_state):
    """询问用户下一步操作"""
    console.print("\n[green]✅ 分析完成！[/green]")
    console.print("[dim]输入 'd' 查看详细结果 | 'n' 运行新分析 | 'q' 退出[/dim]")
    
    while True:
        try:
            user_input = input("请选择操作 (d/n/q): ").strip().lower()
            if user_input == 'd':
                display_detailed_report(final_state)
                console.print("[dim]输入 'n' 运行新分析 | 'q' 退出[/dim]")
            elif user_input == 'n':
                return final_state, "new_analysis"
            elif user_input == 'q':
                return final_state, "quit"
            else:
                console.print("[yellow]无效输入，请输入 'd', 'n' 或 'q'[/yellow]")
        except KeyboardInterrupt:
            console.print("\n[yellow]用户中断，退出...[/yellow]")
            return final_state, "quit"


def display_detailed_report(final_state: Dict):
    """显示详细报告"""
    if not final_state:
        console.print("[red]无结果可显示[/red]")
        return
        
    console.print("\n" + "="*80)
    console.print("[bold blue]ContestTrade 详细分析报告[/bold blue]")
    console.print("="*80)
    
    # 从step_results中获取基本信息
    step_results = final_state.get('step_results', {})
    data_team_results = step_results.get('data_team', {})
    research_team_results = step_results.get('research_team', {})
    
    data_factors_count = data_team_results.get('factors_count', 0)
    research_signals_count = research_team_results.get('signals_count', 0)
    total_events_count = data_team_results.get('events_count', 0) + research_team_results.get('events_count', 0)
    
    # 基本信息
    console.print(f"\n[bold]触发时间:[/bold] {final_state.get('trigger_time', 'N/A')}")
    console.print(f"[bold]数据因子数量:[/bold] {data_factors_count}")
    console.print(f"[bold]研究信号数量:[/bold] {research_signals_count}")
    console.print(f"[bold]总事件数量:[/bold] {total_events_count}")
    
    # 最佳信号
    best_signals = step_results.get('contest', {}).get('best_signals', [])
    if best_signals:
        console.print(f"\n[bold red]🎯 最佳信号 (共{len(best_signals)}个):[/bold red]")
        for i, signal in enumerate(best_signals, 1):
            console.print(f"\n  {i}. {signal.get('symbol_name', 'N/A')} ({signal.get('symbol_code', 'N/A')})")
            console.print(f"     操作: {signal.get('action', 'N/A')}")
            console.print(f"     概率: {signal.get('probability', 'N/A')}")
            console.print(f"     有机会: {signal.get('has_opportunity', 'N/A')}")
            
            # 显示证据详情
            evidence_list = signal.get('evidence_list', [])
            if evidence_list:
                console.print(f"     [bold green]📋 证据详情 (共{len(evidence_list)}个):[/bold green]")
                for j, evidence in enumerate(evidence_list, 1):
                    console.print(f"       {j}. [bold]描述:[/bold] {evidence.get('description', 'N/A')}")
                    console.print(f"          [bold]时间:[/bold] {evidence.get('time', 'N/A')}")
                    console.print(f"          [bold]来源:[/bold] {evidence.get('from_source', 'N/A')}")
                    console.print(f"          [bold]完整描述:[/bold] {evidence.get('description', 'N/A')}")
                    console.print()
            
            # 显示限制条件
            limitations = signal.get('limitations', [])
            if limitations:
                console.print(f"     [bold yellow]⚠️ 限制条件:[/bold yellow]")
                for limitation in limitations:
                    console.print(f"       - {limitation}")
            
            console.print()
    
    console.print("\n" + "="*80)


@app.command()
def run(
    trigger_time: Optional[str] = typer.Option(None, "--time", "-t", help="触发时间 (YYYY-MM-DD HH:MM:SS)"),
    interactive: bool = typer.Option(True, "--interactive/--no-interactive", "-i", help="交互模式"),
):
    """运行ContestTrade分析"""
    
    # 验证配置
    if not validate_config():
        console.print("[red]配置验证失败，请检查配置文件[/red]")
        raise typer.Exit(1)
    
    # 交互模式获取参数
    if interactive:
        if not trigger_time:
            trigger_time = get_trigger_time()
    
    # 验证触发时间
    if not trigger_time:
        console.print("[red]未提供触发时间[/red]")
        raise typer.Exit(1)
    
    try:
        datetime.strptime(trigger_time, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        console.print("[red]触发时间格式错误，请使用 YYYY-MM-DD HH:MM:SS 格式[/red]")
        raise typer.Exit(1)
    
    # 主循环
    while True:
        # 运行分析
        try:
            result = run_contest_analysis_interactive(trigger_time)
        except Exception as e:
            console.print(f"[red]运行分析时发生错误: {e}[/red]")
            break
        
        if result is None or (isinstance(result, tuple) and result[0] is None):
            console.print("[red]❌ 分析失败[/red]")
            break
            
        if isinstance(result, tuple):
            final_state, action = result
            if action == "new_analysis":
                # 用户选择运行新分析
                trigger_time = get_trigger_time()
                continue
            elif action == "quit":
                # 用户选择退出
                break
        else:
            final_state = result
            display = None
        
        # 如果没有明确的下一步动作，就退出
        break
    
    console.print("[green]感谢使用ContestTrade![/green]")


@app.command()
def config():
    """显示当前配置"""
    try:
        if cfg is None:
            console.print("[red]配置模块导入失败[/red]")
            raise typer.Exit(1)
            
        console.print("[bold blue]ContestTrade 配置信息[/bold blue]")
        console.print("="*50)
        
        # LLM配置
        console.print(f"\n[bold]LLM配置:[/bold]")
        console.print(f"  模型: {cfg.llm.get('model_name', 'N/A')}")
        console.print(f"  基础URL: {cfg.llm.get('base_url', 'N/A')}")
        
        # 数据代理配置
        console.print(f"\n[bold]数据代理配置:[/bold]")
        for i, agent_config in enumerate(cfg.data_agents_config, 1):
            console.print(f"  {i}. {agent_config.get('agent_name', 'N/A')}")
            console.print(f"     数据源: {', '.join(agent_config.get('data_source_list', []))}")
        
        # 研究代理配置
        console.print(f"\n[bold]研究代理配置:[/bold]")
        console.print(f"  最大反应步骤: {cfg.research_agent_config.get('max_react_step', 'N/A')}")
        console.print(f"  输出语言: {cfg.research_agent_config.get('output_language', 'N/A')}")
        console.print(f"  工具数量: {len(cfg.research_agent_config.get('tools', []))}")
        
    except Exception as e:
        console.print(f"[red]配置加载失败: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def version():
    """显示版本信息"""
    console.print("[bold blue]ContestTrade[/bold blue]")
    console.print("基于内部竞赛机制的多代理交易系统")
    console.print("Multi-Agent Trading System Based on Internal Contest Mechanism")
    console.print(f"版本: 1.0.0")


if __name__ == "__main__":
    app()