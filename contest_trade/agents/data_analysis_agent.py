""" 

Core Process:
Original Documents → Batch Processing → LLM Intelligent Filtering → Content Deep Summary → Multi-batch Merging → Final Factor

"""
import re
import json
import traceback
import asyncio
import importlib
import pandas as pd
from typing import List, Tuple, Dict, Any, TypedDict
from datetime import datetime
from langgraph.graph import StateGraph, END
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from utils.llm_utils import count_tokens
from models.llm_model import GLOBAL_LLM
from langchain_core.runnables import RunnableConfig
from config.config import PROJECT_ROOT, cfg
from agents.prompts import prompt_for_data_analysis_summary_doc, prompt_for_data_analysis_filter_doc, prompt_for_data_analysis_merge_summary


@dataclass
class DataAnalysisAgentInput:
    """Data Analysis Agent Input"""
    trigger_time: str


@dataclass
class DataAnalysisAgentOutput:
    """Data Analysis Agent Output"""
    agent_name: str
    trigger_time: str
    source_list: List[str]
    bias_goal: str
    context_string: str
    references: List[Dict[str, Any]]
    batch_summaries: List[Dict[str, Any]]

    def to_dict(self):
        return {
            "agent_name": self.agent_name,
            "trigger_time": self.trigger_time,
            "source_list": self.source_list,
            "bias_goal": self.bias_goal,
            "context_string": self.context_string,
            "references": self.references,
            "batch_summaries": self.batch_summaries
        }

@dataclass
class DataAnalysisAgentConfig:
    """Data Analysis Agent Config"""
    agent_name: str
    source_list: List[str]
    max_concurrent_tasks: int
    credits_per_batch: int
    content_cutoff_length: int
    max_llm_context: int
    llm_call_num: int
    final_target_tokens: int
    bias_goal: str = None

    def __init__(
        self,
        agent_name: str = "us_market_summary",
        source_list: List[str] = [],
        max_concurrent_tasks: int = 6,
        credits_per_batch: int = 10,
        content_cutoff_length: int = 2000,
        max_llm_context: int = 28000,
        llm_call_num: int = 2,
        final_target_tokens: int = 4000, 
        bias_goal: str = None,
    ):
        self.agent_name = agent_name
        self.source_list = source_list
        self.max_concurrent_tasks = max_concurrent_tasks
        self.credits_per_batch = credits_per_batch
        self.content_cutoff_length = content_cutoff_length
        self.max_llm_context = max_llm_context
        self.llm_call_num = llm_call_num
        self.final_target_tokens = final_target_tokens
        self.bias_goal = bias_goal
        # Calculate derived parameters based on configuration
        self.batch_count = self.credits_per_batch // self.llm_call_num + 1
        self.title_selection_per_batch = self.max_llm_context // self.content_cutoff_length
        self.summary_target_tokens = self.max_llm_context // self.batch_count


class DataAnalysisAgentState(TypedDict):
    """Detailed Analysis Result Class"""
    trigger_time: str
    source_list: List[str]
    bias_goal: str
    data_source_list: List[Any]
    batch_info: Dict[str, Any]
    batch_results: List[Dict[str, Any]]
    filtered_docs: List[Dict[str, Any]]
    error_log: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    data_df: pd.DataFrame
    summary: str
    processing_stats: Dict[str, Any]
    batch_details: List[Dict[str, Any]]
    result: DataAnalysisAgentOutput


class DataAnalysisAgent:
    """Data Analysis Agent - Records detailed processing procedures"""
    
    def __init__(self, config: DataAnalysisAgentConfig = None):
        self.config = config or DataAnalysisAgentConfig()
        self.app = self._build_graph()

        self.factor_dir = PROJECT_ROOT / "agents_workspace" / "factors" / self.config.agent_name
        if not self.factor_dir.exists():
            self.factor_dir.mkdir(parents=True, exist_ok=True)

        self.set_source_by_config(self.config.source_list)

    def set_source_by_config(self, data_source_list):
        """设置数据源配置"""
        self.data_source_list = []
        
        for source_path in data_source_list:
            try:
                # 解析模块路径和类名
                parts = source_path.split('.')
                class_name = parts[-1]
                module_name = '.'.join(parts[:-1])
                
                # 动态导入模块
                module = importlib.import_module(module_name)
                # 获取类
                data_source_class = getattr(module, class_name)
                
                if not callable(data_source_class):
                    raise ValueError(f"{source_path} is not callable")
                
                # 创建实例
                data_source = data_source_class()
                self.data_source_list.append(data_source)
                print(f"Successfully loaded data source: {source_path}")
                
            except (ImportError, AttributeError) as e:
                print(f"Error loading data source '{source_path}': {e}")
                continue
            except Exception as e:
                print(f"Unexpected error loading '{source_path}': {e}")
                continue


    def _build_graph(self) -> StateGraph:
        """Build the data analysis graph"""
        workflow = StateGraph(DataAnalysisAgentState)
        workflow.add_node("init_factor_dir", self._init_factor_dir)
        workflow.add_node("recompute_factor", self._recompute_factor)
        workflow.add_node("preprocess", self._preprocess)
        workflow.add_node("batch_process", self._batch_process)
        workflow.add_node("final_summary", self._final_summary)
        workflow.add_node("submit_result", self._submit_result)

        workflow.set_entry_point("init_factor_dir")
        workflow.add_conditional_edges("init_factor_dir",
            self._recompute_factor,
            {
                "yes": "preprocess",
                "no": "submit_result"
            })
        workflow.add_edge("recompute_factor", "preprocess")
        workflow.add_edge("preprocess", "batch_process")
        workflow.add_edge("batch_process", "final_summary")
        workflow.add_edge("final_summary", "submit_result")
        workflow.add_edge("submit_result", END)
        return workflow.compile()
    
    async def _init_factor_dir(self, state: DataAnalysisAgentState) -> DataAnalysisAgentState:
        """try to load factor from file"""
        try:
            factor_file = self.factor_dir / f'{state["trigger_time"].replace(" ", "_").replace(":", "-")}.json'
            if factor_file.exists():
                with open(factor_file, 'r', encoding='utf-8') as f:
                    factor_data = json.load(f)
                state["result"] = DataAnalysisAgentOutput(**factor_data)
        except Exception as e:
            print(f"Error loading factor from file: {e}")
            import traceback
            traceback.print_exc()
        return state
    
    async def _recompute_factor(self, state: DataAnalysisAgentState):
        """recompute factor"""
        if state["result"]:
            print(f"Data already exists for {state['trigger_time']}, skipping recompute")
            return "no"
        else:
            print(f"Data does not exist for {state['trigger_time']}, recomputing factor")
            return "yes"

    async def _preprocess(self, state: DataAnalysisAgentState) -> DataAnalysisAgentState:
        """Preprocess the document data"""
        try:
            data_dfs =[]
            for source in self.data_source_list:
                print(f"Getting data from {source.__class__.__name__}...")
                df = await source.get_data(state["trigger_time"])
                required_columns = ['title', 'content', 'pub_time']
                df = df[df['title'].str.strip() != '']
                df = df[df['content'].str.strip() != '']
                df = df[required_columns]
                data_dfs.append(df)
            data_df = pd.concat(data_dfs, ignore_index=True)
            data_df['id'] = range(1, len(data_df) + 1)
            state["data_df"] = data_df

            total_docs = len(data_df)
            batch_count = self.config.batch_count
            batch_size = total_docs // batch_count + 1
            if total_docs % batch_count:
                batch_size += 1
            state["batch_info"] = {
                'batch_count': batch_count,
                'batch_size': batch_size,
                'total_data': total_docs,
                'titles_per_batch': min(self.config.title_selection_per_batch, batch_size)
            }
        except Exception as e:
            print(f"Error preprocessing data: {e}")
            traceback.print_exc()
            return state
        print("preprocess success")
        return state
    
   
    async def _batch_process(self, state: DataAnalysisAgentState) -> DataAnalysisAgentState:
        """Asynchronously process all document batches, return detailed results"""
        print("begin batch process")
        try:
            batch_tasks = []
            for i in range(state["batch_info"]['batch_count']):
                start_idx = i * state["batch_info"]['batch_size']
                end_idx = min((i + 1) * state["batch_info"]['batch_size'], len(state["data_df"]))
                
                if start_idx >= len(state["data_df"]):
                    break
            
                batch_df = state["data_df"].iloc[start_idx:end_idx]
                if not batch_df.empty:
                    batch_tasks.append((i + 1, state["trigger_time"], batch_df, state["batch_info"]['titles_per_batch'], state["bias_goal"]))
        
            # Use semaphore to control concurrency
            semaphore = asyncio.Semaphore(self.config.max_concurrent_tasks)
            
            async def process_single_batch(task_data: Tuple[int, str, pd.DataFrame, int, str]) -> Dict[str, Any]:
                async with semaphore:
                    return await self._process_batch_detailed(task_data)
            
            # Execute all batch tasks
            tasks = [process_single_batch(task) for task in batch_tasks]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Collect results
            batch_results = []
            for result in results:
                if isinstance(result, Exception):
                    batch_results.append({
                        "batch_id": "unknown",
                        "success": False,
                        "error": str(result),
                        "timestamp": datetime.now().isoformat()
                    })
                else:
                    batch_results.append(result)
            
        except Exception as e:
            print(f"Error processing batch: {e}")
            traceback.print_exc()
            batch_results = []
        state["batch_results"] = batch_results
        return state
    

    async def _final_summary(self, state: DataAnalysisAgentState) -> DataAnalysisAgentState:
        """Merge multiple batch summaries into final document factor"""
        try:
            if not state["batch_results"]:
                return "No valid document summaries retrieved"
            
            # Merge all batch summaries
            combined_summary = "\n\n".join([
                f"Batch {i+1} Documents:\n{result.get('summary', '')}" 
                for i, result in enumerate(state["batch_results"])
            ])

            combined_summary_raw = "\n\n".join([
                f"Documents:\n{result.get('summary', '')}" 
                for i, result in enumerate(state["batch_results"])
            ])

            if len(combined_summary_raw) <= self.config.final_target_tokens and not state["bias_goal"]:
                final_summary = combined_summary_raw
            else:
                # Adjust prompt based on whether there's a bias goal
                if state["bias_goal"]:
                    goal_instruction = f"Integrate information around goal '{state['bias_goal']}'"
                    summary_focus = "Highlight important facts related to the goal"
                    final_description = "Final Goal-Oriented Information Summary"
                else:
                    goal_instruction = "Objectively integrate market information"
                    summary_focus = "Maintain objectivity and accuracy of information"
                    final_description = "Final Market Information Summary"
                
                prompt = prompt_for_data_analysis_merge_summary.format(
                    trigger_time=state["trigger_time"],
                    goal_instruction=goal_instruction,
                    combined_summary=combined_summary,
                    summary_focus=summary_focus,
                    final_description=final_description,
                    final_target_tokens=self.config.final_target_tokens,
                    language=cfg.system_language
                )
            
                messages = [{"role": "user", "content": prompt}]
                response = await GLOBAL_LLM.a_run(
                        messages, thinking=False, verbose=False, max_tokens=self.config.final_target_tokens)
                final_summary = response.content.strip()
            
            # Collect all references from batch summaries and final summary
            all_ref_ids = set()
            batch_summaries = []
            
            # Collect references from each batch
            for batch_result in state["batch_results"]:
                if batch_result.get("success") and batch_result.get("summary"):
                    batch_summaries.append({
                        "batch_id": batch_result["batch_id"],
                        "summary": batch_result["summary"],
                        "references": batch_result.get("references", [])
                    })
                    # Add batch reference IDs
                    for ref in batch_result.get("references", []):
                        if isinstance(ref, dict) and "id" in ref:
                            all_ref_ids.add(str(ref["id"]))
            
            # Add final summary references
            final_ref_ids = re.findall(r'\[(\d+)\]', final_summary)
            all_ref_ids.update(final_ref_ids)
            
            # Get all unique references - convert IDs to integers for DataFrame filtering
            try:
                ref_ids_int = [int(ref_id) for ref_id in all_ref_ids if ref_id.isdigit()]
                references_df = state["data_df"][state["data_df"]["id"].isin(ref_ids_int)]
            except (ValueError, TypeError):
                # If conversion fails, try string matching
                references_df = state["data_df"][state["data_df"]["id"].astype(str).isin(all_ref_ids)]
            references = references_df.to_dict(orient="records")
            
            state["summary"] = final_summary
            state["result"] = DataAnalysisAgentOutput(
                agent_name=self.config.agent_name,
                trigger_time=state["trigger_time"],
                source_list=state["source_list"],
                bias_goal=state["bias_goal"],
                context_string=state["summary"],
                references=references,
                batch_summaries=batch_summaries
            )
            return state
        except Exception as e:
            print(f"Error final summary: {e}")
            import traceback
            traceback.print_exc()
            return state

    
    async def _process_batch_detailed(self, task_data: Tuple[int, str, pd.DataFrame, int, str]) -> Dict[str, Any]:
        """Asynchronously process single batch, return detailed results"""
        batch_idx, trigger_datetime, batch_df, titles_to_select, bias_goal = task_data
        batch_start_time = datetime.now()
        
        batch_result = {
            "batch_id": batch_idx,
            "success": False,
            "original_count": len(batch_df),
            "start_time": batch_start_time.isoformat(),
            "filtered_docs": [],
            "summary": "",
            "error": None
        }
        
        print(f"Starting to process batch {batch_idx} ({len(batch_df)} documents)...")
        
        try:
            # Filter document titles
            filtered_df = await self._filter_docs_by_title(trigger_datetime, batch_df, titles_to_select)
            
            # Record filtered document details
            batch_result["filtered_count"] = len(filtered_df)
            batch_result["filtered_docs"] = [
                {
                    "id": row.get('id', ''),
                    "original_index": idx,
                    "title": row.get('title', ''),
                    "pub_time": row.get('pub_time', ''),
                    "content_length": len(str(row.get('content', '')))
                }
                for idx, row in filtered_df.iterrows()
            ]
            
            # Generate content summary
            summary = await self._summarize_doc_content(trigger_datetime, filtered_df, bias_goal)
            
            batch_result["summary"] = summary
            batch_result["summary_length"] = count_tokens(summary) if summary else 0
            batch_result["success"] = True
            
            # Collect references from batch summary
            summary_ref_ids = [int(i) for i in re.findall(r'\[(\d+)\]', summary)]
            batch_result["references"] = filtered_df[filtered_df["id"].isin(summary_ref_ids)].to_dict(orient="records")
            
            print(f"Completed processing batch {batch_idx}")
            
        except Exception as e:
            error_msg = f"Error processing batch {batch_idx}: {e}"
            traceback.print_exc()
            print(error_msg)
            batch_result["error"] = error_msg
        
        # Record batch processing completion time
        batch_end_time = datetime.now()
        batch_result["end_time"] = batch_end_time.isoformat()
        batch_result["processing_duration"] = (batch_end_time - batch_start_time).total_seconds()
        
        return batch_result
    
    
    async def _filter_docs_by_title(self, trigger_datetime: str, batch_df: pd.DataFrame, titles_to_select: int) -> pd.DataFrame:
        """Use LLM to filter most valuable documents based on titles"""
        if batch_df.empty or len(batch_df) <= titles_to_select:
            return batch_df
        
        # Build title context
        titles_context = ""
        for idx, row in batch_df.iterrows():
            doc_id = row.get('id', idx)
            title = row.get('title', '')
            pub_time = row.get('pub_time', '')
            titles_context += f"ID: {doc_id}\nTitle: {title}\nPublish Time: {pub_time}\n\n"
        
        prompt = prompt_for_data_analysis_filter_doc.format(
            trigger_datetime=trigger_datetime,
            titles_to_select=titles_to_select,
            titles_context=titles_context,
            language=cfg.system_language
        )
        
        messages = [{"role": "user", "content": prompt}]
        response = await GLOBAL_LLM.a_run(messages, verbose=False, thinking=False)
        
        # Parse LLM returned IDs
        try:
            selected_ids_str = [x.strip() for x in response.content.strip().split(',') if x.strip()]
            # Try to convert to numbers, if failed keep as string
            selected_ids = []
            for id_str in selected_ids_str:
                try:
                    selected_ids.append(int(id_str))
                except ValueError:
                    selected_ids.append(id_str)
            
            # Filter by id column
            if 'id' in batch_df.columns:
                valid_df = batch_df[batch_df['id'].isin(selected_ids)]
                if not valid_df.empty:
                    return valid_df
            
            return batch_df.head(titles_to_select)
        except:
            return batch_df.head(titles_to_select)
    
    
    async def _summarize_doc_content(self, trigger_datetime: str, batch_df: pd.DataFrame, bias_goal: str = None) -> str:
        """Summarize filtered document content"""
        if batch_df.empty:
            return "No valid document content"
        
        # Build document content context
        doc_context = ""
        doc_raw_content = ""
        for _, row in batch_df.iterrows():
            doc_id = row.get('id', '')
            title = row.get('title', '')
            content = row.get('content', '')
            pub_time = row.get('pub_time', '')
            
            # Truncate content
            if len(content) > self.config.content_cutoff_length:
                content = content[:self.config.content_cutoff_length] + "..."
            
            if pub_time.endswith("23:59:59"):
                pub_time = pub_time.split(" ")[0]
            doc_context += f"<doc id={doc_id}> Title: {title}\nPublish Time: {pub_time}\nContent: {content}</doc>\n"
            doc_raw_content += f"Title: {title}\nPublish Time: {pub_time}\nContent: {content}\n"
        
        if len(doc_context) <= self.config.summary_target_tokens and not bias_goal:
            return doc_raw_content

        # Adjust prompt based on whether there's a bias goal
        if bias_goal:
            bias_instruction = f"Focus on target '{bias_goal}' for targeted summary, emphasizing information related to this goal"
            summary_style = "Goal-oriented Summary"
        else:
            bias_instruction = "Objectively summarize market dynamics and important events"
            summary_style = "Objective Summary"
        
        prompt = prompt_for_data_analysis_summary_doc.format(
            trigger_datetime=trigger_datetime,
            bias_instruction=bias_instruction,
            summary_style=summary_style,
            doc_context=doc_context,
            summary_target_tokens=self.config.summary_target_tokens,
            language=cfg.system_language
        )
        
        messages = [{"role": "user", "content": prompt}]
        response = await GLOBAL_LLM.a_run(messages, verbose=False, max_tokens=self.config.summary_target_tokens)
        
        return response.content.strip()
    
    async def _submit_result(self, state: DataAnalysisAgentState) -> DataAnalysisAgentState:
        """Write the result to a file"""
        try:
            factor_file = self.factor_dir / f'{state["trigger_time"].replace(" ", "_").replace(":", "-")}.json'
            with open(factor_file, 'w', encoding='utf-8') as f:
                json.dump(state["result"].to_dict(), f, ensure_ascii=False, indent=4)
            print(f"Data analysis result saved to {factor_file}")
        except Exception as e:
            print(f"Error writing result: {e}")
            import traceback
            traceback.print_exc()
        return state

    async def run_with_monitoring_events(self, input: DataAnalysisAgentInput, config: RunnableConfig = None) -> DataAnalysisAgentOutput:
        """使用事件流监控运行Agent，返回事件流"""
        initial_state = DataAnalysisAgentState(
            trigger_time=input.trigger_time,
            source_list=self.config.source_list,
            bias_goal=self.config.bias_goal or "",
            data_source_list=[],
            batch_info={},
            batch_results=[],
            filtered_docs=[],
            error_log=[],
            metadata={},
            data_df=pd.DataFrame(),
            summary="",
            processing_stats={},
            batch_details=[],
            result=None
        )
        
        print(f"🚀 Data Analysis Agent Starting - {input.trigger_time}")
        
        # 返回事件流
        async for event in self.app.astream_events(initial_state, version="v2", config=config or RunnableConfig(recursion_limit=50)):
            yield event

    async def run_with_monitoring(self, input: DataAnalysisAgentInput) -> DataAnalysisAgentOutput:
        """使用事件流监控运行Agent"""
        events = self.run_with_monitoring_events(input)
        final_result = None
        async for event in events:
            event_type = event["event"]
            if event_type == "on_chain_start":
                node_name = event["name"]
                if node_name != "__start__":  # 忽略开始事件
                    print(f"🔄 Starting: {node_name}")
            elif event_type == "on_chain_end":
                node_name = event["name"]
                if node_name != "__start__":  # 忽略开始事件
                    print(f"✅ Completed: {node_name}")
                    if node_name == "submit_result":
                        final_state = event.get("data", {}).get("output", None)
                        if final_state and "result" in final_state and final_state["result"]:
                            return final_state["result"]
        return final_result
        

if __name__ == "__main__":
    import json
    # 已删除A股数据源，改用美股数据源进行测试
    from data_source.us_market import USMarket

    data_source_list = [
        "data_source.us_market.USMarket"
    ]
    
    # Create custom configuration
    custom_config = DataAnalysisAgentConfig(
        agent_name="us_market_test",
        source_list=data_source_list,
        final_target_tokens=4000,
        bias_goal="",
    )
    
    # Run detailed analysis generation
    async def main():
        trigger_datetime = "2024-01-23 09:00:00"
        data_agent = DataAnalysisAgent(custom_config)

        agent_input = DataAnalysisAgentInput(
            trigger_time=trigger_datetime,
        )

        output = await data_agent.run_with_monitoring(agent_input)
        print("=== Detailed Analysis Results ===")
        if output and hasattr(output, 'context_string') and output.context_string:
            print(f"Summary: {output.context_string}")
        else:
            print("❌ No summary available", output)
    
    asyncio.run(main()) 


    pass
