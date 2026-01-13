"""
Container management module for high-level orchestration of model containers.
"""

import sys
from typing import Dict, Any, Optional, List

from . import docker_ops
from . import config


def stop_models(model_names: Optional[List[str]] = None) -> None:
    """
    Stop specified models or all models.
    
    Args:
        model_names: List of model names to stop. If None, stops all models.
    """
    existing_containers = docker_ops.get_existing_containers()
    
    if model_names:
        # Stop specific models
        for model_name in model_names:
            container_name = docker_ops.get_model_container_name(model_name)
            if container_name in existing_containers:
                try:
                    docker_ops.stop_container(container_name)
                except docker_ops.DockerError as e:
                    print(f"Error: {e}", file=sys.stderr)
            else:
                print(f"Container {container_name} not found")
    else:
        # Stop all managed containers
        if not existing_containers:
            print("No running inference containers found")
            return
        
        print(f"Stopping {len(existing_containers)} container(s)...")
        for container_name in existing_containers:
            try:
                docker_ops.stop_container(container_name)
            except docker_ops.DockerError as e:
                print(f"Error: {e}", file=sys.stderr)


def down_models(model_names: Optional[List[str]] = None) -> None:
    """
    Stop and remove specified models or all models.
    
    Args:
        model_names: List of model names to remove. If None, removes all models.
    """
    existing_containers = docker_ops.get_existing_containers()
    
    if model_names:
        # Remove specific models
        for model_name in model_names:
            container_name = docker_ops.get_model_container_name(model_name)
            if container_name in existing_containers:
                docker_ops.stop_and_remove_container(container_name)
            else:
                print(f"Container {container_name} not found")
    else:
        # Remove all managed containers
        if not existing_containers:
            print("No inference containers found")
            return
        
        print(f"Removing {len(existing_containers)} container(s)...")
        for container_name in existing_containers:
            docker_ops.stop_and_remove_container(container_name)


def serve_models(model_names: Optional[List[str]] = None, force_reload: bool = False) -> None:
    """
    Serve specified models or all models from config.
    
    Args:
        model_names: List of model names to serve. If None, serves all models from config.
        force_reload: If True, stops and removes existing containers before restarting.
    """
    # Load models configuration
    try:
        models = config.load_models_config()
    except config.ConfigurationError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Build model configs dictionary
    all_model_configs = config.get_all_model_configs(models)
    
    # Determine which models to serve
    if model_names:
        try:
            model_configs = config.validate_model_names(model_names, all_model_configs)
        except config.ConfigurationError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        # Serve all models
        model_configs = all_model_configs
    
    print(f"=== Serving {len(model_configs)} model(s) ===")
    
    # Ensure network exists
    try:
        docker_ops.ensure_network_exists()
    except docker_ops.DockerError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Get existing containers
    existing_containers = docker_ops.get_existing_containers()
    
    # If serving all models, remove containers no longer in config
    if not model_names:
        desired_containers = {docker_ops.get_model_container_name(name) for name in model_configs.keys()}
        to_remove = existing_containers - desired_containers
        for container_name in to_remove:
            docker_ops.stop_and_remove_container(container_name)
    
    # If force reload, remove existing containers that will be restarted
    if force_reload:
        for model_name in model_configs.keys():
            container_name = docker_ops.get_model_container_name(model_name)
            if container_name in existing_containers:
                print(f"Force reload: removing existing container {container_name}")
                docker_ops.stop_and_remove_container(container_name)
        # Refresh the list after removal
        existing_containers = docker_ops.get_existing_containers()
    
    # Start/update containers sequentially to avoid GPU memory race conditions
    for model_name, model_config in model_configs.items():
        container_name = docker_ops.get_model_container_name(model_name)
        
        if container_name in existing_containers:
            print(f"Container {container_name} already exists (skipping)")
        else:
            if docker_ops.start_container(container_name, model_name, model_config):
                # Wait for container to be fully initialized before starting next
                if not docker_ops.wait_for_container_ready(container_name):
                    print(f"Warning: {container_name} may not be fully ready", file=sys.stderr)
    
    print("\n=== Deployment complete ===")


def get_prometheus_config(model_configs: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate Prometheus configuration for the given models.
    
    Args:
        model_configs: Dictionary mapping model names to their configurations
        
    Returns:
        List of Prometheus target configurations
    """
    prometheus_config = []
    for model_name, model_config in model_configs.items():
        container_name = docker_ops.get_model_container_name(model_name)
        prometheus_config.append({
            "targets": [f"{container_name}:8000"],
            "labels": {
                "model": model_name,
                "config": str(model_config)
            }
        })
    return prometheus_config
