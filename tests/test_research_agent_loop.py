import asyncio
import json
import shutil
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTEST_TRADE_ROOT = PROJECT_ROOT / "contest_trade"
if str(CONTEST_TRADE_ROOT) not in sys.path:
    sys.path.insert(0, str(CONTEST_TRADE_ROOT))

from agents.research_agent import ResearchAgent, ResearchAgentConfig
from tools.tool_utils import ToolManager, ToolManagerConfig


class ResearchAgentLoopTests(unittest.TestCase):
    def setUp(self):
        self.agent = ResearchAgent(
            ResearchAgentConfig(agent_name="unit_loop_test", belief="test belief")
        )

    def tearDown(self):
        shutil.rmtree(self.agent.signal_dir, ignore_errors=True)

    def _state(self, tool_call_count, selected_tool):
        return {
            "trigger_time": "2026-05-09 16:26:25",
            "task": "test task",
            "belief": "test belief",
            "background_information": "",
            "plan_result": "",
            "tool_call_context": "",
            "selected_tool": selected_tool,
            "tool_call_count": tool_call_count,
            "step_count": 0,
            "final_result": "",
            "final_result_thinking": "",
            "result": None,
        }

    def test_tool_selection_uses_final_report_after_max_steps(self):
        state = self._state(self.agent.config.max_react_step, {})

        asyncio.run(self.agent._tool_selection(state))

        self.assertEqual({"tool_name": "final_report"}, state["selected_tool"])

    def test_max_steps_takes_precedence_over_tool_selection_error(self):
        state = self._state(
            self.agent.config.max_react_step,
            {"error": "Call tool Failed", "error_msg": "bad tool selection"},
        )

        result = asyncio.run(self.agent._enough_information(state))

        self.assertEqual("enough_information", result)

    def test_tool_selection_error_is_recorded_as_a_tool_step(self):
        state = self._state(
            0,
            {"error": "Call tool Failed", "error_msg": "bad tool selection"},
        )

        asyncio.run(self.agent._call_tool(state))
        recorded_context = json.loads(state["tool_call_context"])

        self.assertEqual(1, state["tool_call_count"])
        self.assertEqual(state["selected_tool"], recorded_context["tool_result"])


class ToolManagerRegistrationTests(unittest.TestCase):
    def test_registers_langchain_structured_tools(self):
        manager = ToolManager(
            ToolManagerConfig(["tools.final_report.final_report"])
        )

        registered_tool = manager.get_tool("final_report")

        self.assertIsNotNone(registered_tool)
        self.assertTrue(hasattr(registered_tool, "invoke"))


if __name__ == "__main__":
    unittest.main()
