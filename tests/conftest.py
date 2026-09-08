# conftest.py — example/ 의 스크립트를 테스트에서 import 할 수 있게 경로를 추가한다
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "example"))
