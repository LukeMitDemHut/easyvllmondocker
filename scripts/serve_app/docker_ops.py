"""
Docker operations module for low-level container management.
"""

import subprocess
import sys
import os
import time
from typing import Set, List, Dict, Any, Optional


class DockerError(Exception):
    """Raised when a Docker operation fails."""
    pass


def get_existing_containers() -> Set[str]:
    """
    Get list of existing inference containers.
    
    Returns:
        Set of container names that are managed by the inference system
    """
    try:
        result = subprocess.run(
            ['docker', 'ps', '-a', '--filter', 'label=inference.managed=true', '--format', '{{.Names}}'],
            capture_output=True,
            text=True,
            check=True
        )
        return set(result.stdout.strip().split('\n')) if result.stdout.strip() else set()
    except subprocess.CalledProcessError:
        return set()


def get_running_containers() -> Set[str]:
    """
    Get list of currently running inference containers.
    
    Returns:
        Set of container names that are running
    """
    try:
        result = subprocess.run(
            ['docker', 'ps', '--filter', 'label=inference.managed=true', '--format', '{{.Names}}'],
            capture_output=True,
            text=True,
            check=True
        )
        return set(result.stdout.strip().split('\n')) if result.stdout.strip() else set()
    except subprocess.CalledProcessError:
        return set()


def get_model_container_name(model_name: str) -> str:
    """
    Get the container name for a model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Container name in the format 'inference-{model_name}'
    """
    return f"inference-{model_name}"


def ensure_network_exists(network_name: str = "inference_default") -> None:
    """
    Ensure the Docker network exists, creating it if necessary.
    
    Args:
        network_name: Name of the Docker network
        
    Raises:
        DockerError: If network creation fails
    """
    try:
        subprocess.run(
            ['docker', 'network', 'inspect', network_name],
            capture_output=True,
            check=True
        )
    except subprocess.CalledProcessError:
        print(f"Creating {network_name} network...")
        try:
            subprocess.run(
                ['docker', 'network', 'create', network_name],
                check=True,
                capture_output=True
            )
            print(f"✓ Network {network_name} created")
        except subprocess.CalledProcessError as e:
            raise DockerError(f"Error creating network: {e}")


def wait_for_container_ready(container_name: str, timeout: int = 500, check_interval: int = 5) -> bool:
    """
    Wait for container to be healthy by checking logs for startup completion.
    
    Args:
        container_name: Name of the container to monitor
        timeout: Maximum time to wait in seconds
        check_interval: How often to check in seconds
        
    Returns:
        True if container is ready, False otherwise
    """
    print(f"Waiting for {container_name} to be ready...", end='', flush=True)
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            # Check if container is still running
            result = subprocess.run(
                ['docker', 'ps', '--filter', f'name={container_name}', '--format', '{{.Names}}'],
                capture_output=True,
                text=True,
                check=True
            )
            
            if container_name not in result.stdout:
                print(f"\n✗ Container {container_name} stopped unexpectedly")
                
                # Check logs for common errors and provide helpful messages
                result = subprocess.run(
                    ['docker', 'logs', container_name],
                    capture_output=True,
                    text=True
                )
                log_output = result.stdout + result.stderr
                
                # Check for KV cache memory error
                if 'KV cache is needed, which is larger than the available KV cache memory' in log_output:
                    import re
                    # Extract suggested max_model_len
                    match = re.search(r'the estimated maximum model length is (\d+)', log_output)
                    if match:
                        suggested_len = match.group(1)
                        print(f"\n⚠️  INSUFFICIENT VRAM FOR KV CACHE")
                        print(f"    Add this to your model configuration in model_config/models.yaml:")
                        print(f"    max-model-len: {suggested_len}")
                    else:
                        print(f"\n⚠️  INSUFFICIENT VRAM FOR KV CACHE")
                        print(f"    Try reducing max-model-len in model_config/models.yaml")
                
                # Check for out of memory error during model loading
                elif 'CUDA out of memory' in log_output and 'Failed to load model' in log_output:
                    print(f"\n⚠️  INSUFFICIENT GPU MEMORY TO LOAD MODEL")
                    print(f"    Try one of the following:")
                    print(f"    1. Reduce gpu-memory-utilization (e.g., 0.70)")
                    print(f"    2. Use tensor-parallel-size with multiple GPUs")
                    print(f"    3. Use a smaller model or stronger quantization")
                
                return False
            
            # Check logs for startup completion message
            result = subprocess.run(
                ['docker', 'logs', container_name],
                capture_output=True,
                text=True
            )
            
            log_output = result.stdout + result.stderr
            if 'Application startup complete' in log_output:
                elapsed = int(time.time() - start_time)
                print(f" ready in {elapsed}s ✓")
                return True
            
            print('.', end='', flush=True)
            time.sleep(check_interval)
            
        except subprocess.CalledProcessError as e:
            print(f"\n✗ Error checking container: {e}")
            return False
    
    print(f"\n✗ Timeout after {timeout}s")
    return False


def stop_container(container_name: str) -> None:
    """
    Stop a Docker container.
    
    Args:
        container_name: Name of the container to stop
        
    Raises:
        DockerError: If stopping fails
    """
    print(f"Stopping {container_name}...")
    try:
        subprocess.run(['docker', 'stop', container_name], check=True, capture_output=True)
        print(f"✓ {container_name} stopped")
    except subprocess.CalledProcessError as e:
        raise DockerError(f"Failed to stop {container_name}: {e}")


def remove_container(container_name: str) -> None:
    """
    Remove a Docker container.
    
    Args:
        container_name: Name of the container to remove
        
    Raises:
        DockerError: If removal fails
    """
    try:
        subprocess.run(['docker', 'rm', container_name], check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        raise DockerError(f"Failed to remove {container_name}: {e}")


def stop_and_remove_container(container_name: str) -> None:
    """
    Stop and remove a container.
    
    Args:
        container_name: Name of the container to stop and remove
    """
    print(f"Removing container: {container_name}")
    try:
        subprocess.run(['docker', 'stop', container_name], check=True, capture_output=True)
        subprocess.run(['docker', 'rm', container_name], check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to remove {container_name}: {e}", file=sys.stderr)


def start_container(
    container_name: str,
    model_name: str,
    model_config: Dict[str, Any],
    network: str = "inference_default",
    port: int = 8000
) -> bool:
    """
    Start a vLLM container for a specific model using docker run.
    
    Args:
        container_name: Name for the Docker container
        model_name: Name of the model
        model_config: Model configuration parameters
        network: Docker network to connect to
        port: Internal port for the vLLM service
        
    Returns:
        True if container started successfully, False otherwise
    """
    print(f"\n--- Starting container for {model_name} ---")
    
    # Extract model parameters
    flat_params = {}
    if isinstance(model_config, dict):
        flat_params = model_config
    else:
        # Handle list format
        for param_dict in model_config:
            if isinstance(param_dict, dict):
                flat_params.update(param_dict)
    
    model_id = flat_params.get('model', model_name)
    gpus = flat_params.get('gpus', None)  # List of GPU IDs or None for all
    
    # Build docker run command
    cmd = [
        'docker', 'run',
        '-d',  # Detached mode
        '--name', container_name,
        '--runtime', 'nvidia',
    ]
    
    # Configure GPU access
    if gpus is None:
        # Use all available GPUs
        cmd.extend(['--gpus', 'all'])
    elif isinstance(gpus, list) and len(gpus) > 0:
        # Use specific GPU(s)
        gpu_list = ','.join(str(gpu) for gpu in gpus)
        cmd.extend(['--gpus', f'"device={gpu_list}"'])
    else:
        # Fallback to all GPUs
        cmd.extend(['--gpus', 'all'])
    
    # Continue with rest of docker run options
    cmd.extend([
        '--ipc', 'host',
        '--network', network,
        '--label', 'inference.managed=true',
        '--label', f'inference.model={model_name}',
        '-v', f'{os.environ.get("HUGGINGFACE_CACHE", os.path.expanduser("~/.cache/huggingface"))}:/root/.cache/huggingface',
    ])
    
    # Add HF_TOKEN if available
    if 'HF_TOKEN' in os.environ:
        cmd.extend(['-e', f'HF_TOKEN={os.environ["HF_TOKEN"]}'])
    
    # Add image and vllm command
    cmd.extend([
        'vllm/vllm-openai:latest',
        model_id,
        '--port', str(port),
        '--host', '0.0.0.0'
    ])
    
    # Add all other parameters as vLLM flags
    for key, value in flat_params.items():
        if key not in ['model', 'gpus']:
            # Convert underscores to hyphens for command-line flags
            flag_name = key.replace('_', '-')
            cmd.extend([f'--{flag_name}', str(value)])
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ Container {container_name} started on port {port}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error starting container for {model_name}: {e}", file=sys.stderr)
        if e.stderr:
            print(f"  {e.stderr}", file=sys.stderr)
        return False


def run_docker_compose(compose_args: List[str], cwd: str) -> None:
    """
    Run a docker compose command.
    
    Args:
        compose_args: Arguments to pass to docker compose
        cwd: Working directory for the command
        
    Raises:
        DockerError: If the command fails
    """
    try:
        subprocess.run(
            ['docker', 'compose'] + compose_args,
            cwd=cwd,
            capture_output=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        raise DockerError(f"Docker compose command failed: {e}")


def docker_exec(container_name: str, command: List[str], input_data: Optional[str] = None) -> subprocess.CompletedProcess:
    """
    Execute a command inside a Docker container.
    
    Args:
        container_name: Name of the container
        command: Command to execute
        input_data: Optional input data to pipe to the command
        
    Returns:
        CompletedProcess object with the command result
        
    Raises:
        DockerError: If the command fails
    """
    docker_cmd = ['docker', 'exec']
    if input_data is not None:
        docker_cmd.append('-i')
    docker_cmd.append(container_name)
    docker_cmd.extend(command)
    
    try:
        return subprocess.run(
            docker_cmd,
            input=input_data,
            text=True if input_data else False,
            check=True,
            capture_output=True
        )
    except subprocess.CalledProcessError as e:
        raise DockerError(f"Docker exec command failed: {e}")
