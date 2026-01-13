"""
Configuration module for loading and managing model configurations.
"""

import yaml
import sys
from pathlib import Path
from typing import List, Dict, Any


class ConfigurationError(Exception):
    """Raised when there is an error in configuration loading or parsing."""
    pass


def load_models_config(config_path: str = "model_config/models.yaml") -> List[Dict[str, Any]]:
    """
    Load the models configuration from YAML file.
    
    Args:
        config_path: Relative path to the configuration file from the project root
        
    Returns:
        List of model configuration dictionaries
        
    Raises:
        ConfigurationError: If the configuration file is not found or invalid
    """
    # Determine the project root (parent of scripts directory)
    script_dir = Path(__file__).parent.parent.parent
    config_file = script_dir / config_path
    
    if not config_file.exists():
        raise ConfigurationError(f"Configuration file {config_file} not found")
    
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Error parsing YAML configuration: {e}")
    
    models = config.get('models', [])
    if not models:
        raise ConfigurationError("No models found in configuration")
    
    return models


def get_all_model_configs(models: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Convert the list of model configurations into a dictionary keyed by model name.
    
    Args:
        models: List of model configuration dictionaries
        
    Returns:
        Dictionary mapping model names to their configurations
    """
    all_model_configs = {}
    all_keys = []
    
    for model_config in models:
        for model_name, params in model_config.items():
            all_keys.append(model_name)
            all_model_configs[model_name] = params
    
    # Check for duplicate keys
    if len(all_keys) != len(all_model_configs):
        duplicate_keys = [key for key in all_keys if all_keys.count(key) > 1]
        unique_duplicates = list(set(duplicate_keys))
        print(f"\n⚠️  WARNING: Duplicate model keys detected in configuration!", file=sys.stderr)
        print(f"    The following keys appear multiple times: {', '.join(unique_duplicates)}", file=sys.stderr)
        print(f"    YAML only keeps the last occurrence. Each model must have a UNIQUE key.", file=sys.stderr)
        print(f"    Expected {len(all_keys)} models but only got {len(all_model_configs)}.", file=sys.stderr)
        print(f"    Use unique suffixes like '-0', '-1', etc. for load-balanced models.\n", file=sys.stderr)
    
    return all_model_configs


def validate_model_names(model_names: List[str], all_model_configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate that requested models exist in configuration.
    
    Args:
        model_names: List of model names to validate
        all_model_configs: Dictionary of all available model configurations
        
    Returns:
        Dictionary of validated model configurations
        
    Raises:
        ConfigurationError: If no valid models are found
    """
    model_configs = {}
    for model_name in model_names:
        if model_name in all_model_configs:
            model_configs[model_name] = all_model_configs[model_name]
        else:
            print(f"Warning: Model '{model_name}' not found in configuration", file=sys.stderr)
    
    if not model_configs:
        raise ConfigurationError("No valid models to serve")
    
    return model_configs


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path to the project root (parent of scripts directory)
    """
    return Path(__file__).parent.parent.parent
