"""
LiteLLM integration module for synchronizing models with the LiteLLM gateway.
"""

import subprocess
import json
import os
import sys
import time
from pathlib import Path
from typing import Set, Dict, Any, Optional, List


class LiteLLMError(Exception):
    """Raised when a LiteLLM operation fails."""
    pass


def get_api_key() -> Optional[str]:
    """
    Get LITELLM_MASTER_KEY from environment or .env file.
    
    Returns:
        API key if found, None otherwise
    """
    # Try environment first
    api_key = os.environ.get('LITELLM_MASTER_KEY')
    if api_key:
        return api_key
    
    # Try .env file
    from . import config
    env_file = config.get_project_root() / '.env'
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                if line.startswith('LITELLM_MASTER_KEY='):
                    return line.split('=', 1)[1].strip()
    
    return None


def get_running_vllm_models() -> Set[str]:
    """
    Get the set of currently running vLLM models.
    
    Returns:
        Set of model names from running containers
    """
    try:
        result = subprocess.run(
            ['docker', 'ps', '--filter', 'label=inference.managed=true', '--format', '{{.Names}}'],
            capture_output=True,
            text=True,
            check=True
        )
        containers = result.stdout.strip().split('\n') if result.stdout.strip() else []
    except subprocess.CalledProcessError:
        print("Warning: Failed to query running containers", file=sys.stderr)
        return set()
    
    # Extract model names from container names using the helper function
    from . import docker_ops
    vllm_models = set()
    for container in containers:
        model_name = docker_ops.get_model_name_from_container(container)
        if model_name:
            vllm_models.add(model_name)
    
    return vllm_models


def get_litellm_models(api_key: str, max_retries: int = 10, initial_delay: float = 1.0) -> List[Dict[str, Any]]:
    """
    Get currently configured models in LiteLLM via its API.
    Retries if the service is not ready yet.
    
    Args:
        api_key: LiteLLM API key
        max_retries: Maximum number of retry attempts (default: 10)
        initial_delay: Initial delay between retries in seconds (default: 1.0)
        
    Returns:
        List of model data dictionaries for models with organization 'inference-vllm'
    """
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            result = subprocess.run(
                ['curl', '-s', '-H', f'Authorization: Bearer {api_key}',
                 'http://localhost:8000/model/info'],
                capture_output=True,
                text=True,
                check=True
            )
            response = json.loads(result.stdout)
            
            # Check if we got an error response (service might be starting up)
            if 'error' in response:
                if attempt < max_retries - 1:
                    print(f"  LiteLLM not ready yet, retrying in {delay:.1f}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    delay *= 1.5  # Exponential backoff
                    continue
                else:
                    print(f"Warning: LiteLLM service error: {response['error']}", file=sys.stderr)
                    return []
            
            litellm_models_data = response.get('data', [])
            
            # Filter for only models with organization "inference-vllm"
            managed_models_data = [
                model for model in litellm_models_data
                if model.get('litellm_params', {}).get('organization') == 'inference-vllm'
            ]
            return managed_models_data
            
        except subprocess.CalledProcessError:
            if attempt < max_retries - 1:
                print(f"  LiteLLM not ready yet, retrying in {delay:.1f}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
                delay *= 1.5  # Exponential backoff
            else:
                print("Warning: Failed to query LiteLLM models after retries", file=sys.stderr)
                return []
        except json.JSONDecodeError as e:
            if attempt < max_retries - 1:
                print(f"  LiteLLM not ready yet, retrying in {delay:.1f}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
                delay *= 1.5  # Exponential backoff
            else:
                print(f"Warning: Failed to parse LiteLM models response: {e}", file=sys.stderr)
                return []
    
    return []


def add_model_to_litellm(model_name: str, hf_model_name: str, api_key: str, max_retries: int = 5, initial_delay: float = 1.0) -> bool:
    """
    Add a model to LiteLLM with retry logic.
    
    Args:
        model_name: Container name of the model  
        hf_model_name: HuggingFace model name (e.g. "openai/gpt-oss-20b")
        api_key: LiteLLM API key
        max_retries: Maximum number of retry attempts (default: 5)
        initial_delay: Initial delay between retries in seconds (default: 1.0)
        
    Returns:
        True if successful, False otherwise
    """
    print(f"Adding model '{model_name}' (HF: {hf_model_name}) to LiteLLM...")
    # Use hosted_vllm provider and actual HF model name
    # Use hf_model_name as the litellm model_name
    from . import docker_ops
    container_name = docker_ops.get_model_container_name(model_name)
    model_payload = {
        "model_name": hf_model_name,
        "litellm_params": {
            "model": f"hosted_vllm/{hf_model_name}",
            "api_base": f"http://{container_name}:8000/v1",
            "organization": "inference-vllm"
        }
    }
    
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            result = subprocess.run(
                ['curl', '-s', '-X', 'POST',
                 '-H', f'Authorization: Bearer {api_key}',
                 '-H', 'Content-Type: application/json',
                 '-d', json.dumps(model_payload),
                 'http://localhost:8000/model/new'],
                capture_output=True,
                text=True,
                check=True
            )
            response = json.loads(result.stdout)
            if 'error' in response:
                # Check if it's a transient error
                error_msg = str(response['error'])
                if attempt < max_retries - 1 and ('connection' in error_msg.lower() or 'timeout' in error_msg.lower()):
                    print(f"  LiteLLM not ready, retrying in {delay:.1f}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    delay *= 1.5
                    continue
                print(f"  ✗ Error adding model: {response['error']}", file=sys.stderr)
                return False
            else:
                print(f"  ✓ Model '{model_name}' added successfully")
                return True
        except subprocess.CalledProcessError as e:
            if attempt < max_retries - 1:
                print(f"  LiteLLM not ready, retrying in {delay:.1f}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
                delay *= 1.5
            else:
                print(f"  ✗ Failed to add model: {e}", file=sys.stderr)
                return False
        except json.JSONDecodeError as e:
            if attempt < max_retries - 1:
                print(f"  LiteLLM not ready, retrying in {delay:.1f}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
                delay *= 1.5
            else:
                print(f"  ✗ Failed to parse response: {e}", file=sys.stderr)
                return False
    
    return False


def remove_model_from_litellm(model_name: str, model_data: Dict[str, Any], api_key: str) -> bool:
    """
    Remove a model from LiteLLM.
    
    Args:
        model_name: Name of the model to remove
        model_data: Full model data from LiteLLM
        api_key: LiteLLM API key
        
    Returns:
        True if successful, False otherwise
    """
    model_id = model_data.get('model_info', {}).get('id')
    if not model_id:
        print(f"  Warning: Cannot remove model '{model_name}', no ID found", file=sys.stderr)
        return False
    
    print(f"Removing model '{model_name}' from LiteLLM...")
    delete_payload = {"id": model_id}
    
    try:
        result = subprocess.run(
            ['curl', '-s', '-X', 'POST',
             '-H', f'Authorization: Bearer {api_key}',
             '-H', 'Content-Type: application/json',
             '-d', json.dumps(delete_payload),
             'http://localhost:8000/model/delete'],
            capture_output=True,
            text=True,
            check=True
        )
        response = json.loads(result.stdout)
        if 'error' in response:
            print(f"  ✗ Error removing model: {response['error']}", file=sys.stderr)
            return False
        else:
            print(f"  ✓ Model '{model_name}' removed successfully")
            return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Failed to remove model: {e}", file=sys.stderr)
        return False
    except json.JSONDecodeError as e:
        print(f"  ✗ Failed to parse response: {e}", file=sys.stderr)
        return False


def update_litellm() -> None:
    """
    Update LiteLLM gateway configuration to sync with running vLLM containers.
    """
    print("\nSynchronizing models with LiteLLM gateway...")
    
    # Load model config to get HF model names
    from . import config, docker_ops
    try:
        models = config.load_models_config()
        all_model_configs = config.get_all_model_configs(models)
    except config.ConfigurationError as e:
        print(f"Warning: Failed to load model config: {e}", file=sys.stderr)
        all_model_configs = {}
    
    # Get running vLLM models (container model names)
    vllm_models = get_running_vllm_models()
    print(f"Found {len(vllm_models)} vLLM model(s): {', '.join(sorted(vllm_models)) if vllm_models else 'none'}")
    
    # Build expected container names for running models
    expected_containers = {docker_ops.get_model_container_name(name) for name in vllm_models}
    
    # Get API key
    api_key = get_api_key()
    if not api_key:
        print("Warning: LITELLM_MASTER_KEY not found, skipping model sync", file=sys.stderr)
        return
    
    # Get LiteLLM models (returns list of all instances)
    litellm_models_list = get_litellm_models(api_key)
    print(f"Found {len(litellm_models_list)} managed LiteLLM instance(s)")
    
    # Build set of api_base URLs already registered in LiteLLM
    # Extract container name from api_base URL (e.g., "http://inference-model:8000/v1" -> "inference-model")
    registered_containers = set()
    for model_data in litellm_models_list:
        api_base = model_data.get('litellm_params', {}).get('api_base', '')
        # Extract container name from URL: http://CONTAINER:8000/v1
        if api_base.startswith('http://'):
            container_name = api_base.split('://')[1].split(':')[0]
            registered_containers.add(container_name)
    
    # Add models present in vLLM containers but missing in LiteLLM
    containers_to_add = expected_containers - registered_containers
    models_to_add = {docker_ops.get_model_name_from_container(c) for c in containers_to_add if docker_ops.get_model_name_from_container(c)}
    for model_name in models_to_add:
        # Get HF model name from config
        model_config = all_model_configs.get(model_name, {})
        hf_model_name = model_config.get('model', model_name)
        add_model_to_litellm(model_name, hf_model_name, api_key)
    
    # Remove LiteLLM instances whose containers are not running
    for model_data in litellm_models_list:
        api_base = model_data.get('litellm_params', {}).get('api_base', '')
        if api_base.startswith('http://'):
            container_name = api_base.split('://')[1].split(':')[0]
            if container_name not in expected_containers:
                # This instance points to a non-existent container
                remove_model_from_litellm(container_name, model_data, api_key)
    
    stale_containers = registered_containers - expected_containers
    if not models_to_add and not stale_containers:
        print("✓ Models are in sync")
    else:
        print(f"✓ Sync complete: {len(models_to_add)} added, {len(stale_containers)} removed")
