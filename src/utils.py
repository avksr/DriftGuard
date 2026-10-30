"""
Utility functions for DriftGuard.
Shared configuration loading and helper routines.
"""

import os
from pathlib import Path
from typing import Any, Dict, Union
import yaml


def load_config(config_path: Union[str, Path] = "config.yaml") -> Dict[str, Any]:
    """
    Load project configuration from a YAML file.
    
    Args:
        config_path: Path or string pointing to config.yaml.
        
    Returns:
        Dictionary of configuration parameters.
    """
    p = Path(config_path)
    if not p.is_file():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
