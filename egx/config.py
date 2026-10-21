import yaml
from pathlib import Path

def load_config(path=None):
    path = Path(path) if path else Path(__file__).resolve().parent.parent / "config" / "settings.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
