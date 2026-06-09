from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
DATASET_PATH = DATA_DIR / "dataset.txt"


def resolve_project_path(path: Path) -> Path:
  if path.is_absolute():
    return path
  return PROJECT_ROOT / path
