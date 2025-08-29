import sys
from pathlib import Path

contest_home = Path(__file__).parent.parent / 'contest_trade'
sys.path.insert(0, str(contest_home))# ContestTrade CLI Package 