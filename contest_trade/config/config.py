"""
config module for trade agent
"""
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).parent.parent.resolve()


class ProjectConfig:

    def __init__(self) -> None:
        yaml_path = PROJECT_ROOT.parent / "config_us.yaml"

        with open(yaml_path, "r", encoding="utf-8") as fr:
            config = yaml.load(fr, Loader=yaml.FullLoader)
        for k in config:
            setattr(self, k, config[k])

cfg = ProjectConfig()

if __name__ == "__main__":
    print(cfg.data_agents_config)
    print(cfg.research_agent_config)
    print(cfg.llm)
    print(cfg.llm_thinking)
    print(cfg.vlm)