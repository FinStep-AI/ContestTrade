"""
ContestTrade Final Report Template
最终报告模板生成器
"""
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.markdown import Markdown
from rich.layout import Layout
from rich.live import Live
from rich.align import Align
from rich import box
import re

class DataReportGenerator:
    """数据报告生成器"""
    
    def __init__(self, factors_data: Dict):
        self.factors_data = factors_data
        self.console = Console()
        
    def generate_markdown_report(self, save_path: Path) -> str:
        """生成数据报告的Markdown格式"""
        
        # 获取触发时间
        trigger_time = self.factors_data.get('trigger_time', 'N/A')
        
        # 统计数据源数量和代理数量
        total_agents = len(self.factors_data.get('agents', {}))
        
        report_content = f"""# ContestTrade 数据分析报告

## 📊 数据摘要

**分析时间**: {trigger_time}  
**分析状态**: ✅ 完成  
**数据代理数量**: {total_agents}  

---

## 🔍 数据源分析详情

"""
        
        # 遍历每个代理的数据
        for agent_name, agent_data in self.factors_data.get('agents', {}).items():
            report_content += f"### 📈 {agent_name.replace('_', ' ').title()}\n\n"
            
            # 只获取context_string字段
            context_string = agent_data.get('context_string', '')
            
            if context_string:
                # 清洗掉 [Batch X] 标记
                cleaned_context = re.sub(r'\[Batch \d+\]', '', context_string).strip()
                report_content += f"{cleaned_context}\n\n"
            else:
                report_content += "**暂无分析内容**\n\n"
            
            report_content += "---\n\n"
        
        # 免责声明
        report_content += "## ⚠️ 免责声明\n\n"
        report_content += "本报告由ContestTrade数据分析系统生成，数据来源于各个数据代理的分析结果，仅供参考。\n\n"
        report_content += f"**报告生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report_content += f"**系统版本**: ContestTrade v1.1.0\n"
        
        # 保存到文件
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        return report_content
    
    def display_terminal_interactive_report(self, markdown_content: str):
        """显示可滚动的交互式终端数据报告"""
        
        # 创建Rich控制台，启用可滚动功能
        console = Console()
        
        # 创建Markdown对象
        markdown = Markdown(markdown_content)
        
        # 创建面板
        report_panel = Panel(
            markdown,
            title="📋 ContestTrade 数据分析报告",
            title_align="center",
            border_style="blue",
            padding=(1, 2),
        )
        
        # 清屏并显示报告
        console.clear()
        console.print(report_panel)
        
        # 操作提示
        console.print(f"\n[yellow]📖 报告查看说明:[/yellow]")
        console.print(f"[dim]• 向上滚动查看报告开头内容[/dim]")
        console.print(f"[dim]• 向下滚动查看更多详细信息[/dim]") 
        console.print(f"[dim]• 按任意键返回主菜单[/dim]")
        
        try:
            input()
        except KeyboardInterrupt:
            pass
    
    def display_interactive_report(self, markdown_content: str, save_path: Path):
        """显示可滚动的交互式数据报告"""
        
        # 创建Rich控制台，启用可滚动功能
        console = Console()
        
        # 创建Markdown对象
        markdown = Markdown(markdown_content)
        
        report_panel = Panel(
            markdown,
            title="📋 ContestTrade Data Report",
            title_align="center",
            border_style="blue",
            padding=(1, 2),
        )
        
        # 清屏并显示报告
        console.clear()
        console.print(report_panel)
        
        # 显示文件保存信息和操作提示
        console.print(f"\n[green]✅ 数据报告已保存至:[/green]")
        console.print(f"[blue]📄 {save_path}[/blue]")
        console.print(f"[dim]您可以使用文本编辑器打开查看完整报告[/dim]")
        
        # 操作提示
        console.print(f"\n[yellow]📖 报告操作说明:[/yellow]")
        console.print(f"[dim]• 向上滚动查看报告开头[/dim]")
        console.print(f"[dim]• 向下滚动查看更多内容[/dim]") 
        console.print(f"[dim]• 按任意键返回主菜单[/dim]")
        
        try:
            input()
        except KeyboardInterrupt:
            pass


class FinalReportGenerator:
    """最终报告生成器"""
    
    def __init__(self, final_state: Dict):
        self.final_state = final_state
        self.console = Console()
        
    def generate_markdown_report(self, save_path: Path) -> str:
        """生成Markdown格式的报告"""
        
        # 获取基本信息
        step_results = self.final_state.get('step_results', {})
        data_team_results = step_results.get('data_team', {})
        research_team_results = step_results.get('research_team', {})
        contest_results = step_results.get('contest', {})
        
        # 获取触发时间，确保正确解析
        trigger_time = self.final_state.get('trigger_time', 'N/A')
        if trigger_time == 'N/A':
            # 尝试从其他地方获取时间
            trigger_time = step_results.get('trigger_time', 'N/A')
        
        data_factors_count = data_team_results.get('factors_count', 0)
        research_signals_count = research_team_results.get('signals_count', 0)
        best_signals = contest_results.get('best_signals', [])
        
        # 筛选有效信号
        valid_signals = [s for s in best_signals if s.get('has_opportunity', 'no') == 'yes']
        invalid_signals = [s for s in best_signals if s.get('has_opportunity', 'no') != 'yes']
        
        # 生成报告内容
        signal_rate = f"{len(valid_signals)/len(best_signals)*100:.1f}% ({len(valid_signals)}/{len(best_signals)})" if len(best_signals) > 0 else "0% (0/0)"
        
        report_content = f"""# ContestTrade 最终分析报告

## 📊 执行摘要

**分析时间**: {trigger_time}  
**分析状态**: ✅ 完成  
**数据源数量**: {data_factors_count}  
**研究信号数量**: {research_signals_count}  
**有效投资信号**: {len(valid_signals)}  
**信号有效率**: {signal_rate}

---

## 🎯 投资建议摘要

"""
        
        if valid_signals:
            report_content += f"### ✅ 推荐投资信号 ({len(valid_signals)}个)\n\n"
            
            for i, signal in enumerate(valid_signals, 1):
                symbol_name = signal.get('symbol_name', 'N/A')
                symbol_code = signal.get('symbol_code', 'N/A')
                action = signal.get('action', 'N/A')
                agent_id = signal.get('agent_id', 'N/A')
                
                report_content += f"#### {i}. {symbol_name} ({symbol_code})\n\n"
                report_content += f"- **投资动作**: {action}\n"
                report_content += f"- **分析来源**: Research Agent {agent_id}\n"
                
                # 证据详情
                evidence_list = signal.get('evidence_list', [])
                if evidence_list:
                    report_content += f"- **支撑证据** ({len(evidence_list)}项):\n"
                    for j, evidence in enumerate(evidence_list, 1):
                        desc = evidence.get('description', 'N/A')
                        source = evidence.get('from_source', 'N/A')
                        time = evidence.get('time', 'N/A')
                        report_content += f"  {j}. **{desc}** (来源: {source}, 时间: {time})\n"
                
                # 风险提示
                limitations = signal.get('limitations', [])
                if limitations:
                    report_content += f"- **风险提示**:\n"
                    for limitation in limitations:
                        report_content += f"  - {limitation}\n"
                
                report_content += "\n"
        else:
            report_content += "### ❌ 暂无推荐投资信号\n\n"
            report_content += "本次分析未发现具有明确投资机会的信号。\n\n"
        
        # 无效信号统计
        if invalid_signals:
            report_content += f"### ⚠️ 排除信号 ({len(invalid_signals)}个)\n\n"
            report_content += "以下信号经分析后认为不具备投资机会：\n\n"
            
            for i, signal in enumerate(invalid_signals, 1):
                agent_id = signal.get('agent_id', 'N/A')
                report_content += f"{i}. Research Agent {agent_id} - 无明确投资机会\n"
            
            report_content += "\n"
        
        # 免责声明
        report_content += "---\n\n## ⚠️ 免责声明\n\n"
        report_content += "本报告由ContestTrade AI系统生成，仅供参考，不构成投资建议。投资有风险，决策需谨慎。\n\n"
        report_content += f"**报告生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report_content += f"**系统版本**: ContestTrade v1.0.0\n"
        
        # 保存到文件
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        return report_content
    
    def display_terminal_interactive_report(self, markdown_content: str):
        """显示可滚动的交互式终端报告（不保存文件）"""
        
        # 创建Rich控制台，启用可滚动功能
        console = Console()
        
        # 创建Markdown对象
        markdown = Markdown(markdown_content)
        
        # 创建面板
        report_panel = Panel(
            markdown,
            title="📋 ContestTrade 详细分析报告",
            title_align="center",
            border_style="blue",
            padding=(1, 2),
        )
        
        # 清屏并显示报告
        console.clear()
        console.print(report_panel)
        
        # 操作提示
        console.print(f"\n[yellow]📖 报告查看说明:[/yellow]")
        console.print(f"[dim]• 向上滚动查看报告开头内容[/dim]")
        console.print(f"[dim]• 向下滚动查看更多详细信息[/dim]") 
        console.print(f"[dim]• 按任意键返回主菜单[/dim]")
        
        try:
            input()
        except KeyboardInterrupt:
            pass
    
    def display_interactive_report(self, markdown_content: str, save_path: Path):
        """显示可滚动的交互式报告"""
        
        # 创建Rich控制台，启用可滚动功能
        console = Console()
        
        # 创建Markdown对象
        markdown = Markdown(markdown_content)
        
        report_panel = Panel(
            markdown,
            title="📋 ContestTrade Final Report",
            title_align="center",
            border_style="blue",
            padding=(1, 2),
        )
        
        # 清屏并显示报告
        console.clear()
        console.print(report_panel)
        
        # 显示文件保存信息和操作提示
        console.print(f"\n[green]✅ 报告已保存至:[/green]")
        console.print(f"[blue]📄 {save_path}[/blue]")
        console.print(f"[dim]您可以使用文本编辑器打开查看完整报告[/dim]")
        
        # 操作提示
        console.print(f"\n[yellow]📖 报告操作说明:[/yellow]")
        console.print(f"[dim]• 向上滚动查看报告开头[/dim]")
        console.print(f"[dim]• 向下滚动查看更多内容[/dim]") 
        console.print(f"[dim]• 按任意键返回主菜单[/dim]")
        
        try:
            input()
        except KeyboardInterrupt:
            pass
    
    def create_summary_table(self) -> Table:
        """创建摘要表格"""
        table = Table(title="投资信号摘要", box=box.ROUNDED)
        
        table.add_column("序号", style="cyan", no_wrap=True)
        table.add_column("股票名称", style="magenta")
        table.add_column("股票代码", style="magenta")
        table.add_column("投资动作", style="green")
        table.add_column("分析来源", style="blue")
        table.add_column("状态", style="yellow")
        
        step_results = self.final_state.get('step_results', {})
        best_signals = step_results.get('contest', {}).get('best_signals', [])
        
        for i, signal in enumerate(best_signals, 1):
            symbol_name = signal.get('symbol_name', 'N/A')
            symbol_code = signal.get('symbol_code', 'N/A')
            action = signal.get('action', 'N/A')
            agent_id = signal.get('agent_id', 'N/A')
            has_opportunity = signal.get('has_opportunity', 'no')
            
            status = "✅ 推荐" if has_opportunity == 'yes' else "❌ 排除"
            
            table.add_row(
                str(i),
                symbol_name,
                symbol_code,
                action,
                f"Agent {agent_id}",
                status
            )
        
        return table

def generate_data_report(factors_data: Dict, results_dir: Path) -> tuple[str, Path]:
    """生成数据报告"""
    
    # 创建数据报告生成器
    generator = DataReportGenerator(factors_data)
    
    # 生成文件名
    trigger_time = factors_data.get('trigger_time', 'N/A')
    
    if trigger_time != 'N/A' and trigger_time is not None:
        safe_time = trigger_time.replace(' ', '_').replace(':', '-')
    else:
        safe_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    filename = f"data_report_{safe_time}.md"
    data_reports_dir = results_dir / "data_reports"
    data_reports_dir.mkdir(parents=True, exist_ok=True)
    save_path = data_reports_dir / filename
    
    markdown_content = generator.generate_markdown_report(save_path)
    
    return markdown_content, save_path


def display_data_report_interactive(factors_data: Dict, results_dir: Path):
    """显示交互式数据报告"""
    
    markdown_content, save_path = generate_data_report(factors_data, results_dir)
    generator = DataReportGenerator(factors_data)
    generator.display_interactive_report(markdown_content, save_path)
    
    return save_path


def generate_final_report(final_state: Dict, results_dir: Path) -> tuple[str, Path]:
    """生成最终报告"""
    
    # 创建报告生成器
    generator = FinalReportGenerator(final_state)
    
    # 生成文件名
    trigger_time = final_state.get('trigger_time', 'N/A')
    
    if trigger_time != 'N/A' and trigger_time is not None:
        safe_time = trigger_time.replace(' ', '_').replace(':', '-')
    
    filename = f"final_report_{safe_time}.md"
    research_reports_dir = results_dir / "research_reports"
    research_reports_dir.mkdir(parents=True, exist_ok=True)
    save_path = research_reports_dir / filename
    markdown_content = generator.generate_markdown_report(save_path)
    
    return markdown_content, save_path


def display_final_report_interactive(final_state: Dict, results_dir: Path):
    """显示交互式最终报告"""

    markdown_content, save_path = generate_final_report(final_state, results_dir)
    generator = FinalReportGenerator(final_state)
    generator.display_interactive_report(markdown_content, save_path)
    
    return save_path
