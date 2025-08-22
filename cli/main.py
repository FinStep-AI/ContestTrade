"""
ContestTrade: 基于内部竞赛机制的Multi-Agent交易系统
"""
import asyncio
import sys
import json
import re
from pathlib import Path
from typing import Optional, Dict
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

from .utils import get_trigger_time, validate_required_services
from .static.report_template import display_final_report_interactive
from contest_trade.config.config import cfg, PROJECT_ROOT
sys.path.append(str(PROJECT_ROOT))
from contest_trade.main import SimpleTradeCompany
from contest_trade.utils.tushare_utils import get_trade_date
from contest_trade.models.llm_model import GLOBAL_LLM

console = Console()

app = typer.Typer(
    name="contesttrade",
    help="ContestTrade: 基于内部竞赛机制的Multi-Agent交易系统",
    add_completion=True,
)

def _get_agent_config():
    """从配置文件动态获取代理配置"""
    agent_status = {}
    
    # 从配置文件获取数据代理
    data_agents_config = cfg.data_agents_config
    for agent_config in data_agents_config:
        agent_name = agent_config.get('agent_name', '')
        if agent_name:
            agent_status[agent_name] = "pending"
    
    # 从belief_list.json获取研究代理数量
    belief_list_path = PROJECT_ROOT / "config" / "belief_list.json"

    with open(belief_list_path, 'r', encoding='utf-8') as f:
        belief_list = json.load(f)
    # 根据belief数量创建研究代理
    for i in range(len(belief_list)):
        agent_status[f"agent_{i}"] = "pending"
    
    return agent_status
class ContestTradeDisplay:
    """ContestTrade显示管理器"""
    
    def __init__(self):
        self.messages = deque(maxlen=200)  # 增加消息队列容量
        self.agent_status = _get_agent_config()
        self.current_task = "初始化系统..."
        self.progress_info = ""
        self.final_state = None
        self.analysis_completed = False
        self.step_counts = {"data": 0, "research": 0, "contest": 0, "finalize": 0}
        self._last_update_hash = None  # 用于检测内容是否真正发生变化
        self._last_console_size = None  # 用于检测控制台大小变化
        
        # 日志监控相关
        self.logs_dir = Path(PROJECT_ROOT) / "agents_workspace" / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
    def create_log_file(self, trigger_time: str):
        """创建本次运行的日志文件"""
        timestamp = trigger_time.replace(":", "-").replace(" ", "_")
        self.log_file = self.logs_dir / f"run_{timestamp}.log"
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write(f"ContestTrade Run Log - {trigger_time}\n")
            f.write("=" * 50 + "\n")
        
    def check_agent_status_from_events_and_files(self, trigger_time: str):
        """基于事件和文件系统更新agent状态"""
        # 格式化时间戳用于文件匹配
        timestamp_str = trigger_time.replace("-", "-").replace(":", "-").replace(" ", "_")
        
        # 检查factors目录（Data Analysis Agent结果）
        factors_dir = Path(PROJECT_ROOT) / "agents_workspace" / "factors"
        if factors_dir.exists():
            for agent_name in self.agent_status:
                if not agent_name.startswith("agent_"):  # Data agents
                    agent_dir = factors_dir / agent_name
                    if agent_dir.exists():
                        # 查找对应时间戳的文件
                        pattern = f"{timestamp_str}*.json"
                        files = list(agent_dir.glob(pattern))
                        if files and self.agent_status[agent_name] != "completed":
                            self.update_agent_status(agent_name, "completed")
                            self.add_message("Data Analysis Agent", f"✅ {agent_name} 完成数据分析")
        
        # 检查reports目录（Research Agent结果）
        reports_dir = Path(PROJECT_ROOT) / "agents_workspace" / "reports"
        if reports_dir.exists():
            for agent_name in self.agent_status:
                if agent_name.startswith("agent_"):  # Research agents
                    agent_dir = reports_dir / agent_name
                    if agent_dir.exists():
                        # 查找对应时间戳的文件
                        pattern = f"{timestamp_str}*.json"
                        files = list(agent_dir.glob(pattern))
                        if files and self.agent_status[agent_name] != "completed":
                            self.update_agent_status(agent_name, "completed")
                            self.add_message("Research Agent", f"✅ {agent_name} 完成研究分析")
    
    def start_data_agents(self):
        """开始所有Data Analysis Agent"""
        for agent_name in self.agent_status:
            if not agent_name.startswith("agent_"):  # Data agents
                self.update_agent_status(agent_name, "running")
        self.add_message("系统", "🚀 开始运行所有Data Analysis Agent")
    
    def start_research_agents(self):
        """开始所有Research Agent"""
        for agent_name in self.agent_status:
            if agent_name.startswith("agent_"):  # Research agents
                self.update_agent_status(agent_name, "running")
        self.add_message("系统", "🚀 开始运行所有Research Agent")
        
    def add_message(self, message_type: str, content: str):
        """添加消息"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        new_message = f"[{timestamp}] {message_type}: {content}"
        self.messages.append(new_message)
        
    def should_update_display(self) -> bool:
        """检查是否需要更新显示（内容是否发生变化）"""
        current_hash = hash(str(self.messages) + self.current_task + self.progress_info + str(self.agent_status))
        if current_hash != self._last_update_hash:
            self._last_update_hash = current_hash
            return True
        return False
    
    def console_size_changed(self) -> bool:
        """检查控制台大小是否发生变化"""
        current_size = console.size
        if current_size != self._last_console_size:
            self._last_console_size = current_size
            return True
        return False
        
    def update_agent_status(self, agent_name: str, status: str):
        """更新Agent状态"""
        if agent_name not in self.agent_status:
            self.agent_status[agent_name] = "pending"
            
        self.agent_status[agent_name] = status
        
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
        
        # 获取终端大小
        console_size = console.size
        
        # 根据终端高度调整header大小
        header_size = min(10, max(9, console_size.height // 6))
        
        layout.split_column(
            Layout(name="header", size=header_size),
            Layout(name="main_content")
        )
        
        # 根据终端宽度调整左右面板比例
        if console_size.width < 120:
            left_ratio, right_ratio = 2, 3  # 窄屏时调整比例
        else:
            left_ratio, right_ratio = 4, 7  # 宽屏时的比例
            
        layout["main_content"].split_row(
            Layout(name="left_panel", ratio=left_ratio),
            Layout(name="right_panel", ratio=right_ratio)
        )
        layout["left_panel"].split_column(
            Layout(name="status", ratio=3),
            Layout(name="progress", ratio=2)
        )
        layout["right_panel"].split_column(
            Layout(name="content", ratio=3),
            Layout(name="footer", ratio=2)
        )
        
        return layout
        
    def update_display(self, layout: Layout, trigger_time: str):
        """更新显示"""
        welcome_text = Path(__file__).parent / "static" / "welcome.txt"
        if welcome_text.exists():
            with open(welcome_text, "r", encoding="utf-8") as f:
                welcome = f.read()
        else:
            welcome = "ContestTrade: 基于内部竞赛机制的Multi-Agent交易系统"
        
        header_panel = Panel(
            Align.center(welcome),
            title="🎯 ContestTrade - 基于内部竞赛机制的Multi-Agent交易系统",
            border_style="blue",
            padding=(0, 1),
            expand=True  # 自适应宽度
        )
        layout["header"].update(header_panel)
        
        # 更新Agent状态面板
        status_text = Text()
        
        # Data Analysis Agent状态
        data_agents = {k: v for k, v in self.agent_status.items() if not k.startswith("agent_")}
        if data_agents:
            status_text.append("📊 Data Analysis Agent\n", style="bold cyan")
            for agent_name, status in data_agents.items():
                status_icon = {
                    "pending": "⏳等待中...",
                    "running": "🔄分析中...", 
                    "completed": "✅分析完成"
                }.get(status, "❓")
                
                agent_display = agent_name[:20].ljust(20)
                status_text.append(f"{agent_display} {status_icon}\n")
        
        # Research Agent状态
        research_agents = {k: v for k, v in self.agent_status.items() if k.startswith("agent_")}
        if research_agents:
            status_text.append("\n🔍 Research Agent\n", style="bold green")
            for agent_name, status in research_agents.items():
                status_icon = {
                    "pending": "⏳等待中...",
                    "running": "🔄分析中...", 
                    "completed": "✅分析完成"
                }.get(status, "❓")
                
                agent_display = agent_name[:20].ljust(20)
                status_text.append(f"{agent_display} {status_icon}\n")
        
        status_panel = Panel(
            status_text,
            title="🤖 Agent状态",
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
        progress_text.append(f"  Data Analysis Agent事件: {self.step_counts['data']}\n")
        progress_text.append(f"  Research Agent事件: {self.step_counts['research']}\n")
        # progress_text.append(f"  竞赛事件: {self.step_counts['contest']}\n")
        # progress_text.append(f"  完成事件: {self.step_counts['finalize']}\n")
        
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
            for msg in list(self.messages)[-8:]:
                content_text.append(f"{msg}\n")
        else:
            content_text.append("  ⏳ 等待事件...\n")
        
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
            footer_text.append("🔄 分析进行中...预计等待10分钟...", style="bold yellow")
            if self.analysis_completed:
                footer_text.append("\n✅ 分析完成！请按回车键(↵)退出运行界面...", style="bold green")
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
            
            summary_text.append(f"📊 数据源: {data_factors_count} | ", style="green")
            summary_text.append(f"🔍 研究信号: {research_signals_count} | ", style="blue")
            
            # 获取所有信号并筛选有机会的信号
            best_signals = step_results.get('contest', {}).get('best_signals', [])
            
            # 筛选 has_opportunity 为 yes 的信号
            valid_signals = []
            for signal in best_signals:
                has_opportunity = signal.get('has_opportunity', 'no')
                if has_opportunity == 'yes':
                    valid_signals.append(signal)
            
            if valid_signals:
                summary_text.append(f"🎯 有效信号: {len(valid_signals)}", style="bold red")
                
                for i, signal in enumerate(valid_signals):
                    symbol_name = signal.get('symbol_name', 'N/A')
                    action = signal.get('action', 'N/A')
                    agent_id = signal.get('agent_id', 'N/A')
                    
                    summary_text.append(f"\n  {i+1}. Research Agent{agent_id}：", style="yellow")
                    summary_text.append(f"{symbol_name}({action}) ", style="cyan")
                    
            else:
                summary_text.append("🎯 有效信号: 0", style="bold red")     
                summary_text.append(" 无有效信号")

            summary_text.append("\n💡分析完成，按回车退出运行界面...")
        else:
            summary_text.append("❌ 分析失败", style="red")
        
        return summary_text


def run_contest_analysis_interactive(trigger_time: str):
    """在交互界面中运行竞赛分析"""
    try:
        # 创建显示管理器
        display = ContestTradeDisplay()
        
        # 创建初始布局
        layout = display.create_layout(trigger_time)
        
        # 使用Live界面运行 - 提高刷新频率以更好响应窗口大小变化
        with Live(layout, refresh_per_second=4, screen=True, auto_refresh=True, console=console) as live:
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
            final_state = asyncio.run(run_with_events_capture(company, trigger_time, display, layout, live))
            
            # 运行结束后
            if final_state:
                display.add_message("完成", "✅ 分析完成！")
                display.set_current_task("分析完成，生成报告...")
                display.set_analysis_completed(True)
                display.final_state = final_state
                display.update_display(layout, trigger_time)
                
                # 自动生成MD报告
                try:
                    results_dir = Path(PROJECT_ROOT) / "agents_workspace" / "results"
                    from .static.report_template import generate_final_report, generate_data_report
                    
                    # 生成研究报告
                    markdown_content, report_path = generate_final_report(final_state, results_dir)
                    display.add_message("报告", f"✅ 研究报告已生成: {report_path.name}")
                    
                    # 生成数据报告
                    factors_data = load_factors_data(trigger_time)
                    if factors_data and factors_data.get('agents'):
                        data_markdown_content, data_report_path = generate_data_report(factors_data, results_dir)
                        display.add_message("报告", f"✅ 数据报告已生成: {data_report_path.name}")
                    else:
                        display.add_message("报告", f"⚠️ 未找到数据源，跳过数据报告生成")
                    
                    display.update_display(layout, trigger_time)
                except Exception as e:
                    display.add_message("报告", f"⚠️ MD报告生成失败: {str(e)}")
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


async def run_with_events_capture(company, trigger_time: str, display: ContestTradeDisplay, layout, live):
    """运行公司工作流并捕获事件流"""
    try:
        display.add_message("开始", "🚀 开始运行工作流...")
        display.set_current_task("🔄 启动工作流...")
        display.create_log_file(trigger_time)
        display.update_display(layout, trigger_time)
        
        # 启动定期检查文件状态的任务
        async def periodic_status_check():
            while not display.analysis_completed:
                display.check_agent_status_from_events_and_files(trigger_time)
                
                # 检查控制台大小是否变化，如果变化则重新创建布局
                if display.console_size_changed():
                    new_layout = display.create_layout(trigger_time)
                    # 将新布局的内容复制到当前布局中
                    layout.update(new_layout)
                    display.update_display(layout, trigger_time)
                else:
                    display.update_display(layout, trigger_time)
                    
                await asyncio.sleep(1)  # 每1秒检查一次，提高响应性
        
        # 启动状态检查任务
        status_check_task = asyncio.create_task(periodic_status_check())
        
        # 运行公司工作流并处理事件
        final_state = None
        async for event in company.run_company_with_events(trigger_time):
            event_name = event.get("name", "")
            event_type = event.get("event", "")
            event_data = event.get("data", {})
            
            # 记录重要事件到日志
            if event_type in ["on_chain_start", "on_chain_end"]:
                log_msg = f"[{datetime.now().strftime('%H:%M:%S')}] {event_type}: {event_name}\n"
                with open(display.log_file, "a", encoding="utf-8") as f:
                    f.write(log_msg)
                # # 同时显示到界面事件流
                # display.add_message("事件", f"{event_type}: {event_name}")
            
            # 记录自定义事件到日志和界面
            if event_type == "on_custom":
                custom_event_name = event_name
                custom_data = event_data
                log_msg = f"[{datetime.now().strftime('%H:%M:%S')}] CUSTOM: {custom_event_name} - {custom_data}\n"
                with open(display.log_file, "a", encoding="utf-8") as f:
                    f.write(log_msg)
                # 显示到界面
                display.add_message("自定义事件", f"{custom_event_name}")
            
            # 处理stdout输出（记录到日志和界面）
            if event_type == "on_stdout":
                stdout_content = event_data.get("chunk", "")
                if stdout_content.strip():
                    log_msg = f"[{datetime.now().strftime('%H:%M:%S')}] STDOUT: {stdout_content.strip()}\n"
                    with open(display.log_file, "a", encoding="utf-8") as f:
                        f.write(log_msg)
                    # 显示所有stdout到界面
                    display.add_message("输出", stdout_content.strip())
            
            # 处理关键阶段事件
            if event_type == "on_chain_start":
                stage_config = {
                    "run_data_agents": {
                        "action": display.start_data_agents,
                        "task": "🔄 Data Analysis Agent 数据收集阶段",
                        "progress": "数据收集阶段 1/4"
                    },
                    "run_research_agents": {
                        "action": display.start_research_agents,
                        "task": "🔄 Research Agent 研究分析阶段", 
                        "progress": "研究分析阶段 2/4"
                    },
                    "run_contest": {
                        "action": lambda: None,
                        "task": "🔄 竞赛评选阶段",
                        "progress": "竞赛评选阶段 3/4"
                    },
                    "finalize": {
                        "action": lambda: None,
                        "task": "🔄 结果生成阶段",
                        "progress": "结果生成阶段 4/4"
                    }
                }
                
                if event_name in stage_config:
                    config = stage_config[event_name]
                    config["action"]()
                    display.set_current_task(config["task"])
                    display.set_progress_info(config["progress"])
            
            # 处理完成事件
            elif event_type == "on_chain_end":
                completion_config = {
                    "run_data_agents": {
                        "task": "✅ Data Analysis Agent 完成",
                        "message": "✅ 所有Data Analysis Agent完成"
                    },
                    "run_research_agents": {
                        "task": "✅ Research Agent 完成", 
                        "message": "✅ 所有Research Agent完成"
                    },
                    "run_contest": {
                        "task": "✅ 竞赛评选完成",
                        "message": None
                    },
                    "finalize": {
                        "task": "✅ 结果生成完成",
                        "message": None,
                        "special": True
                    }
                }
                
                if event_name in completion_config:
                    config = completion_config[event_name]
                    display.set_current_task(config["task"])
                    if config.get("message"):
                        display.add_message("系统", config["message"])
                    
                    if config.get("special"):  # finalize阶段的特殊处理
                        final_state = event_data.get("output", {})
                        if 'trigger_time' not in final_state:
                            final_state['trigger_time'] = trigger_time
                        display.set_analysis_completed(True)
            
            # 处理具体的节点事件（用于步骤统计）
            if event_type == "on_chain_start":
                step_mapping = {
                    "data": ["init_factor", "recompute_factor", "submit_result", "preprocess", "batch_process", "final_summary"],
                    "research": ["init_signal", "recompute_signal", "init_data", "plan", "tool_selection", "call_tool", "write_result"],
                    "contest": ["run_contest", "run_judger_critic"],
                    "finalize": ["finalize"]
                }
                
                for step_type, keywords in step_mapping.items():
                    if any(keyword in event_name.lower() for keyword in keywords):
                        display.step_counts[step_type] += 1
                        break
            
            # 更新显示 - 由于启用了自动刷新，不需要手动refresh
            display.update_display(layout, trigger_time)
        
        # 停止状态检查任务并设置最终状态
        if 'status_check_task' in locals():
            status_check_task.cancel()
            try:
                await status_check_task
            except asyncio.CancelledError:
                pass
        
        # 设置所有Agent为完成状态
        for agent_name in display.agent_status:
            display.update_agent_status(agent_name, "completed")
        
        # 确保final_state包含trigger_time
        if final_state is not None and 'trigger_time' not in final_state:
            final_state['trigger_time'] = trigger_time
        
        return final_state
        
    except Exception as e:
        # 停止状态检查任务
        if 'status_check_task' in locals():
            status_check_task.cancel()
            try:
                await status_check_task
            except asyncio.CancelledError:
                pass
        
        display.add_message("错误", f"❌ 运行失败: {str(e)}")
        console.print(f"[red]详细错误: {e}[/red]")
        return None


def ask_user_for_next_action(final_state):
    """询问用户下一步操作"""
    console.print("\n[green]✅ 分析完成！[/green]")
    console.print("[dim]输入 'rr' 查看研究报告 | 'dr' 查看数据报告 | 'n' 运行新分析 | 'q' 退出[/dim]")
    
    while True:
        try:
            user_input = input("请选择操作 (rr/dr/n/q): ").strip().lower()
            if user_input == 'rr':
                display_detailed_report(final_state)
                console.print("[dim]输入 'rr' 查看研究报告 | 'dr' 查看数据报告 | 'n' 运行新分析 | 'q' 退出[/dim]")
            elif user_input == 'dr':
                display_data_report(final_state)
                console.print("[dim]输入 'rr' 查看研究报告 | 'dr' 查看数据报告 | 'n' 运行新分析 | 'q' 退出[/dim]")
            elif user_input == 'n':
                return final_state, "new_analysis"
            elif user_input == 'q':
                return final_state, "quit"
            else:
                console.print("[yellow]无效输入，请输入 'rr', 'dr', 'n' 或 'q'[/yellow]")
        except KeyboardInterrupt:
            console.print("\n[yellow]用户中断，退出...[/yellow]")
            return final_state, "quit"

def display_data_report(final_state: Dict):
    """显示数据分析报告"""
    if not final_state:
        console.print("[red]无结果可显示[/red]")
        return
    
    try:
        from .static.report_template import DataReportGenerator
        
        # 从final_state获取trigger_time，然后读取factors数据
        trigger_time = final_state.get('trigger_time', 'N/A')
        
        # 读取factors文件夹中的数据
        factors_data = load_factors_data(trigger_time)
        
        if not factors_data or not factors_data.get('agents'):
            console.print("[yellow]未找到数据分析结果[/yellow]")
            return
        
        generator = DataReportGenerator(factors_data)
        
        # 生成报告内容
        total_agents = len(factors_data.get('agents', {}))
        
        markdown_content = f"""# ContestTrade 数据分析报告

## 📊 数据摘要

**分析时间**: {trigger_time}  
**分析状态**: ✅ 完成  
**数据代理数量**: {total_agents}  

---

## 🔍 数据源分析详情

"""
        
        # 遍历每个代理的数据
        for agent_name, agent_data in factors_data.get('agents', {}).items():
            markdown_content += f"### 📈 {agent_name.replace('_', ' ').title()}\n\n"
            
            # 只获取context_string字段
            context_string = agent_data.get('context_string', '')
            
            if context_string:
                # 清洗掉 [Batch X] 标记
                cleaned_context = re.sub(r'\[Batch \d+\]', '', context_string).strip()
                markdown_content += f"{cleaned_context}\n\n"
            else:
                markdown_content += "**暂无分析内容**\n\n"
            
            markdown_content += "---\n\n"
        
        generator.display_terminal_interactive_report(markdown_content)
        
    except Exception as e:
        console.print(f"[red]数据报告显示失败: {e}[/red]")
        console.print("[yellow]正在显示简化版数据报告...[/yellow]")
        
        # 显示简化版数据报告
        try:
            factors_data = load_factors_data(final_state.get('trigger_time', 'N/A'))
            if factors_data and factors_data.get('agents'):
                console.print(f"\n[bold]数据分析摘要:[/bold]")
                console.print(f"数据代理数量: {len(factors_data.get('agents', {}))}")
                
                for agent_name in factors_data.get('agents', {}):
                    console.print(f"- {agent_name}")
            else:
                console.print("[yellow]未找到数据分析结果[/yellow]")
        except Exception as inner_e:
            console.print(f"[red]简化版数据报告也显示失败: {inner_e}[/red]")


def load_factors_data(trigger_time: str) -> Dict:
    """加载factors文件夹中的数据"""
    factors_data = {
        'trigger_time': trigger_time,
        'agents': {}
    }
    
    # 格式化时间戳用于文件匹配
    if trigger_time and trigger_time != 'N/A':
        timestamp_str = trigger_time.replace("-", "-").replace(":", "-").replace(" ", "_")
    else:
        return factors_data
    
    # 读取factors目录
    factors_dir = Path(PROJECT_ROOT) / "agents_workspace" / "factors"
    if not factors_dir.exists():
        return factors_data
    
    try:
        for agent_dir in factors_dir.iterdir():
            if agent_dir.is_dir():
                agent_name = agent_dir.name
                
                # 查找对应时间戳的JSON文件
                pattern = f"{timestamp_str}*.json"
                files = list(agent_dir.glob(pattern))
                
                if files:
                    # 读取第一个匹配的文件
                    with open(files[0], 'r', encoding='utf-8') as f:
                        agent_data = json.load(f)
                        factors_data['agents'][agent_name] = agent_data
    except Exception as e:
        console.print(f"[yellow]加载factors数据时出错: {e}[/yellow]")
    
    return factors_data


def display_detailed_report(final_state: Dict):
    """显示详细的可滚动终端报告（使用Rich交互式显示）"""
    if not final_state:
        console.print("[red]无结果可显示[/red]")
        return
    
    try:
        from .static.report_template import FinalReportGenerator
        generator = FinalReportGenerator(final_state)
        step_results = final_state.get('step_results', {})
        data_team_results = step_results.get('data_team', {})
        research_team_results = step_results.get('research_team', {})
        contest_results = step_results.get('contest', {})
        
        trigger_time = final_state.get('trigger_time', 'N/A')
        data_factors_count = data_team_results.get('factors_count', 0)
        research_signals_count = research_team_results.get('signals_count', 0)
        best_signals = contest_results.get('best_signals', [])
        
        valid_signals = [s for s in best_signals if s.get('has_opportunity', 'no') == 'yes']
        invalid_signals = [s for s in best_signals if s.get('has_opportunity', 'no') != 'yes']
        
        signal_rate = f"{len(valid_signals)/len(best_signals)*100:.1f}% ({len(valid_signals)}/{len(best_signals)})" if len(best_signals) > 0 else "0% (0/0)"
        
        markdown_content = f"""# ContestTrade 详细分析报告

## 📊 执行摘要

**分析时间**: {trigger_time}  
**数据源数量**: {data_factors_count}  
**研究信号数量**: {research_signals_count}  
**有效投资信号**: {len(valid_signals)}  
**信号有效率**: {signal_rate}

---

## 🎯 投资信号详情
"""
        
        if valid_signals:
            markdown_content += f"\n### ✅ 推荐投资信号 ({len(valid_signals)}个)\n\n"
            
            for i, signal in enumerate(valid_signals, 1):
                symbol_name = signal.get('symbol_name', 'N/A')
                symbol_code = signal.get('symbol_code', 'N/A')
                action = signal.get('action', 'N/A')
                probability = signal.get('probability', 'N/A')
                agent_id = signal.get('agent_id', 'N/A')
                
                markdown_content += f"#### {i}. {symbol_name} ({symbol_code})\n\n"
                markdown_content += f"- **投资动作**: {action}\n"
                markdown_content += f"- **分析来源**: Research Agent {agent_id}\n\n"
                
                evidence_list = signal.get('evidence_list', [])
                if evidence_list:
                    markdown_content += f"**📋 支撑证据 ({len(evidence_list)}项):**\n\n"
                    for j, evidence in enumerate(evidence_list, 1):
                        desc = evidence.get('description', 'N/A')
                        source = evidence.get('from_source', 'N/A')
                        time = evidence.get('time', 'N/A')
                        markdown_content += f"{j}. **{desc}**\n"
                        markdown_content += f"   - 时间: {time}\n"
                        markdown_content += f"   - 来源: {source}\n\n"
                
                # 风险提示
                limitations = signal.get('limitations', [])
                if limitations:
                    markdown_content += f"**⚠️ 潜在风险:**\n\n"
                    for limitation in limitations:
                        markdown_content += f"- {limitation}\n"
                    markdown_content += "\n"
                
                markdown_content += "---\n"
        else:
            markdown_content += "\n### ❌ 暂无推荐投资信号\n\n"
            markdown_content += "本次分析未发现具有明确投资机会的信号。\n\n"
        
        # 无效信号统计
        if invalid_signals:
            markdown_content += f"### ⚠️ 排除信号 ({len(invalid_signals)}个)\n"
            markdown_content += "以下信号经分析后认为不具备投资机会：\n\n"
            
            for i, signal in enumerate(invalid_signals, 1):
                agent_id = signal.get('agent_id', 'N/A')
                markdown_content += f"{i}. Research Agent {agent_id} - 无明确投资机会\n"
            
            markdown_content += "\n"
        generator.display_terminal_interactive_report(markdown_content)
        
    except Exception as e:
        console.print(f"[red]交互式报告显示失败: {e}[/red]")
        console.print("[yellow]正在显示简化版报告...[/yellow]")
        
        # 显示简化版报告
        step_results = final_state.get('step_results', {})
        best_signals = step_results.get('contest', {}).get('best_signals', [])
        valid_signals = [s for s in best_signals if s.get('has_opportunity', 'no') == 'yes']
        
        console.print(f"\n[bold]分析摘要:[/bold]")
        console.print(f"总信号: {len(best_signals)}, 有效信号: {len(valid_signals)}")
        
        for i, signal in enumerate(valid_signals, 1):
            console.print(f"{i}. {signal.get('symbol_name', 'N/A')} - {signal.get('action', 'N/A')}")

@app.command()
def run(
    trigger_time: Optional[str] = typer.Option(None, "--time", "-t", help="触发时间 (YYYY-MM-DD HH:MM:SS)"),
):
    """运行ContestTrade分析"""

    # 获取触发时间
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
    
    # 验证必需的服务连接（自动根据tushare_key决定验证策略）
    if not validate_required_services():
        console.print("[red]系统验证失败，无法启动分析[/red]")
        raise typer.Exit(1)
    
    # 主循环
    while True:
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
                trigger_time = get_trigger_time()
                if not validate_required_services():
                    console.print("[red]系统验证失败，无法启动分析[/red]")
                    break
                continue
            elif action == "quit":
                break
        else:
            final_state = result
            display = None

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
        
        console.print(f"\n[bold]LLM配置:[/bold]")
        console.print(f"  模型: {cfg.llm.get('model_name', 'N/A')}")
        console.print(f"  基础URL: {cfg.llm.get('base_url', 'N/A')}")
        
        # Data Analysis Agent配置
        console.print(f"\n[bold]Data Analysis Agent配置:[/bold]")
        for i, agent_config in enumerate(cfg.data_agents_config, 1):
            console.print(f"  {i}. {agent_config.get('agent_name', 'N/A')}")
            console.print(f"     数据源: {', '.join(agent_config.get('data_source_list', []))}")
        
        # Research Agent配置
        console.print(f"\n[bold]Research Agent配置:[/bold]")
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
    console.print("基于内部竞赛机制的Multi-Agent交易系统")
    console.print("Multi-Agent Trading System Based on Internal Contest Mechanism")
    console.print(f"版本: 1.1")

if __name__ == "__main__":
    app()