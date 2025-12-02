import yaml
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

def load_panel(panel_name: str) -> Dict[str, Any]:
    """Load a specific panel configuration from YAML file."""
    panel_path = Path(__file__).parent / "panels" / f"{panel_name}.yaml"
    if not panel_path.exists():
        raise ValueError(f"Panel configuration not found: {panel_name}")
    
    with open(panel_path, "r", encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_all_panels() -> Dict[str, Dict[str, Any]]:
    """Load all panel configurations from the panels directory."""
    panels_dir = Path(__file__).parent / "panels"
    panels = {}
    for path in panels_dir.glob("*.yaml"):
        try:
            panels[path.stem] = load_panel(path.stem)
        except Exception as e:
            logger.error(f"Failed to load panel {path.stem}: {e}")
    return panels

def load_human() -> Dict[str, Any]:
    """Load human configuration from YAML file."""
    human_path = Path(__file__).parent / "human.yaml"
    with open(human_path, "r", encoding='utf-8') as f:
        return yaml.safe_load(f) 