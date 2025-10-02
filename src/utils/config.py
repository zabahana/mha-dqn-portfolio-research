"""
Configuration utilities for MHA-DQN Portfolio Optimization Research
"""

import yaml
import os
from typing import Dict, Any
from pathlib import Path


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: Dict[str, Any], config_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration file
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def get_device_config() -> str:
    """
    Get optimal device configuration.
    
    Returns:
        Device string ('cuda', 'mps', or 'cpu')
    """
    import torch
    
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def update_config_paths(config: Dict[str, Any], base_path: str = '.') -> Dict[str, Any]:
    """
    Update relative paths in configuration to absolute paths.
    
    Args:
        config: Configuration dictionary
        base_path: Base path for relative paths
        
    Returns:
        Updated configuration dictionary
    """
    base_path = Path(base_path).resolve()
    
    # Update common path fields
    path_fields = ['data_dir', 'model_dir', 'results_dir', 'log_dir']
    
    for field in path_fields:
        if field in config:
            path = Path(config[field])
            if not path.is_absolute():
                config[field] = str(base_path / path)
    
    return config
