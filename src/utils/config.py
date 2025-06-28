import json
import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file"""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    if config_path.suffix in ['.yml', '.yaml']:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    elif config_path.suffix == '.json':
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {config_path.suffix}")
    
    return config


def save_config(config: Dict[str, Any], config_path: str):
    """Save configuration to file"""
    config_path = Path(config_path)
    
    if config_path.suffix in ['.yml', '.yaml']:
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    elif config_path.suffix == '.json':
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    else:
        raise ValueError(f"Unsupported config format: {config_path.suffix}")