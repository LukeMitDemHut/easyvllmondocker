"""
Prometheus integration module for managing Prometheus monitoring configuration.
"""

import json
import sys
from typing import List, Dict, Any

from . import docker_ops
from . import config


class PrometheusError(Exception):
    """Raised when a Prometheus operation fails."""
    pass


def update_prometheus_config(prometheus_config: List[Dict[str, Any]]) -> None:
    """
    Update Prometheus configuration with current model targets.
    
    Args:
        prometheus_config: List of Prometheus target configurations
        
    Raises:
        PrometheusError: If configuration update fails
    """
    try:
        # Ensure prometheus container is running using docker compose
        project_root = config.get_project_root()
        docker_ops.run_docker_compose(['up', '-d', 'prometheus'], str(project_root))

        # Write configuration to Prometheus container
        json_data = json.dumps(prometheus_config, indent=2)
        
        docker_ops.docker_exec(
            'prometheus',
            ['sh', '-c', 'cat > /etc/prometheus/vllm_targets.json'],
            input_data=json_data
        )
        
        # Reload Prometheus configuration
        docker_ops.docker_exec(
            'prometheus',
            ['kill', '-HUP', '1']
        )
        
        print("✓ Prometheus configuration updated and reloaded")

    except docker_ops.DockerError as e:
        print(f"Warning: Failed to update Prometheus: {e}", file=sys.stderr)
    except Exception as e:
        print(f"Warning: Failed to update Prometheus: {e}", file=sys.stderr)


def get_prometheus_targets(model_configs: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate Prometheus target configurations for the given models.
    
    Args:
        model_configs: Dictionary mapping model names to their configurations
        
    Returns:
        List of Prometheus target configurations
    """
    prometheus_config = []
    for model_name, model_config in model_configs.items():
        prometheus_config.append({
            "targets": [f"inference-{model_name}:8000"],
            "labels": {
                "model": model_name,
                "config": str(model_config)
            }
        })
    return prometheus_config
