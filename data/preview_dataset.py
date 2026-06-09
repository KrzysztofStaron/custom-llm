import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from paths import DATASET_PATH

with DATASET_PATH.open("r", encoding="utf-8", errors="ignore") as f:
    for i, line in enumerate(f):
        if i >= 200:
            break
        print(line, end='')
