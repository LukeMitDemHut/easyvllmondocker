"""
Docker operations module for low-level container management.
"""

import subprocess
import sys
import os
import time
import re
import shlex
import socket
import datetime
from typing import Set, List, Dict, Any, Optional


class DockerError(Exception):
    """Raised when a Docker operation fails."""
    pass


def is_offline() -> bool:
    """
    Detect whether the device is offline (air-gapped).

    Returns True if the SERVE_OFFLINE environment variable is set to a
    truthy value, or if a TCP connection to huggingface.co:443 fails
    within a short timeout (indicating no internet access).
    """
    if os.environ.get('SERVE_OFFLINE', '').lower() in ('1', 'true', 'yes'):
        return True
    try:
        socket.create_connection(('huggingface.co', 443), timeout=3)
        return False
    except (socket.timeout, OSError):
        return True


def _redact_command(cmd: List[str]) -> str:
    """
    Format a command for display while hiding sensitive environment values.
    """
    redacted = []
    for token in cmd:
        if token.startswith('HF_TOKEN='):
            redacted.append('HF_TOKEN=<redacted>')
        else:
            redacted.append(token)
    return shlex.join(redacted)


def print_container_logs_hint(container_name: str) -> None:
    """
    Print a ready-to-run command for investigating container startup issues.
    """
    print("    Inspect the container logs with:")
    print(f"    docker logs {shlex.quote(container_name)}")


def stop_failed_startup_container(container_name: str) -> None:
    """
    Stop a failed startup container so restart policies do not keep retrying it.
    """
    try:
        subprocess.run(['docker', 'stop', container_name], check=False, capture_output=True)
        print(f"    Stopped {container_name} to prevent restart loops.")
    except Exception as e:
        print(f"    Warning: failed to stop {container_name}: {e}", file=sys.stderr)


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
    
    # detect illegal characters in model_name
    if not re.match(r'^[a-zA-Z0-9\-_.\/]+$', model_name):
        raise ValueError(f"Model name '{model_name}' contains illegal characters. Allowed: a-z, A-Z, 0-9, -, _, ., /")
    
    # prohibit double hyphens and double periods
    if '--' in model_name or '..' in model_name:
        raise ValueError(f"Model name '{model_name}' contains prohibited sequences. Double hyphens (--) and double periods (..) are not allowed.")
    
    # tranlate slashes to double hyphens since HF uses them to separate org/model and docker does not allow slashes in container names
    model_name = model_name.replace('/', '--')
    return f"inference-{model_name}"


def get_model_name_from_container(container_name: str) -> Optional[str]:
    """
    Extract the model name from a container name, reversing the transformation.
    
    Args:
        container_name: Name of the container (e.g., 'inference-openai--gpt-oss-120b-0')
        
    Returns:
        Model name with slashes restored (e.g., 'openai/gpt-oss-120b-0'), or None if invalid format
    """
    if not container_name.startswith('inference-'):
        return None
    
    # Remove 'inference-' prefix
    model_name = container_name[len('inference-'):]
    
    # Reverse the transformation: double hyphens back to slashes
    # But only if they're not at the start/end and not triple hyphens
    model_name = model_name.replace('--', '/')
    
    return model_name


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


def get_container_runtime_state(container_name: str) -> Optional[Dict[str, Any]]:
    """
    Get runtime state details for a container.

    Args:
        container_name: Name of the container

    Returns:
        Dict with runtime state fields, or None if container cannot be inspected.
    """
    try:
        result = subprocess.run(
            [
                'docker', 'inspect', '--format',
                '{{.State.Running}}|{{.State.Restarting}}|{{.RestartCount}}|{{.State.StartedAt}}|{{.State.Status}}|{{.State.ExitCode}}|{{.State.Error}}',
                container_name,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        return None

    output = result.stdout.strip()
    if not output:
        return None

    # Split at most 6 times to preserve any '|' that may appear in the error string.
    parts = output.split('|', 6)
    if len(parts) < 7:
        return None

    running_raw, restarting_raw, restart_count_raw, started_at, status, exit_code_raw, error_msg = parts

    try:
        restart_count = int(restart_count_raw)
    except ValueError:
        restart_count = 0

    try:
        exit_code = int(exit_code_raw)
    except ValueError:
        exit_code = -1

    return {
        'running': running_raw.lower() == 'true',
        'restarting': restarting_raw.lower() == 'true',
        'restart_count': restart_count,
        'started_at': started_at,
        'status': status,
        'exit_code': exit_code,
        'error': error_msg,
    }


def wait_for_container_ready(container_name: str, timeout: int = None, check_interval: int = 5) -> bool:
    """
    Wait for container to be healthy by checking logs for startup completion.
    
    Args:
        container_name: Name of the container to monitor
        timeout: Maximum time to wait in seconds (None for no timeout)
        check_interval: How often to check in seconds
        
    Returns:
        True if container is ready, False otherwise
    """
    print(f"Waiting for {container_name} to be ready...", end='', flush=True)
    start_time = time.time()
    # Small lookback avoids missing very early startup lines emitted right before this wait begins.
    log_since_dt = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=15)
    log_since = log_since_dt.isoformat(timespec='seconds').replace('+00:00', 'Z')

    initial_state = get_container_runtime_state(container_name)
    if initial_state is None:
        print(f"\n✗ Container {container_name} could not be inspected")
        print_container_logs_hint(container_name)
        return False

    initial_restart_count = initial_state['restart_count']
    initial_started_at = initial_state['started_at']
    
    while timeout is None or time.time() - start_time < timeout:
        try:
            state = get_container_runtime_state(container_name)
            if state is None or (not state['running'] and not state['restarting']):
                print(f"\n✗ Container {container_name} stopped unexpectedly")
                print_container_logs_hint(container_name)
                
                # Check logs for common errors and provide helpful messages
                result = subprocess.run(
                    ['docker', 'logs', container_name],
                    capture_output=True,
                    text=True
                )
                log_output = result.stdout + result.stderr
                
                # Check for KV cache memory error
                if 'KV cache is needed, which is larger than the available KV cache memory' in log_output:
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

                # Check for vLLM startup memory reservation error
                elif 'Free memory on device' in log_output and 'is less than desired GPU memory utilization' in log_output:
                    match = re.search(
                        r'Free memory on device (?P<device>\S+) '
                        r'\((?P<free>[\d.]+)/(?P<total>[\d.]+) GiB\) '
                        r'on startup is less than desired GPU memory utilization '
                        r'\((?P<util>[\d.]+), (?P<desired>[\d.]+) GiB\)',
                        log_output
                    )
                    print(f"\n⚠️  GPU MEMORY UTILIZATION TOO HIGH FOR CURRENT FREE MEMORY")
                    if match:
                        free_gib = float(match.group('free'))
                        total_gib = float(match.group('total'))
                        suggested_util = max(0.01, (free_gib / total_gib) - 0.02)
                        print(f"    {match.group('device')} has {free_gib:.2f}/{total_gib:.2f} GiB free at startup.")
                        print(f"    Requested utilization {match.group('util')} needs about {float(match.group('desired')):.2f} GiB.")
                        print(f"    Set gpu-memory-utilization below {free_gib / total_gib:.2f}; for example:")
                        print(f"    gpu-memory-utilization: {suggested_util:.2f}")
                    else:
                        print(f"    Reduce gpu-memory-utilization in model_config/models.yaml")
                        print(f"    or stop other GPU processes before starting this model.")
                    print(f"    You can also reduce max-model-len or stop other GPU workloads.")
                
                # Check for out of memory error during model loading
                elif 'CUDA out of memory' in log_output and 'Failed to load model' in log_output:
                    print(f"\n⚠️  INSUFFICIENT GPU MEMORY TO LOAD MODEL")
                    print(f"    Try one of the following:")
                    print(f"    1. Reduce gpu-memory-utilization (e.g., 0.70)")
                    print(f"    2. Use tensor-parallel-size with multiple GPUs")
                    print(f"    3. Use a smaller model or stronger quantization")

                # Check for openai_harmony vocab download/cache error (common for GPT-OSS)
                elif 'openai_harmony.HarmonyError: error downloading or loading vocab file' in log_output:
                    print(f"\n⚠️  OPENAI_HARMONY VOCAB DOWNLOAD/CACHE ERROR")
                    print(f"    Detected failure while loading GPT-OSS harmony encoding files.")
                    print(f"    Ensure the container has a persistent tiktoken cache.")
                    print(f"    This project mounts host cache to /tmp/tiktoken-rs-cache via TIKTOKEN_RS_CACHE_DIR.")
                    print(f"    If first startup happened without this mount, run with --force-reload.")
                
                stop_failed_startup_container(container_name)
                return False

            if state['restart_count'] > initial_restart_count:
                print(
                    f"\n✗ Container {container_name} restarted during startup "
                    f"(restart count {initial_restart_count} -> {state['restart_count']})"
                )
                print_container_logs_hint(container_name)
                if state['error']:
                    print(f"    Last Docker error: {state['error']}")
                stop_failed_startup_container(container_name)
                return False

            if state['started_at'] != initial_started_at:
                print(
                    f"\n✗ Container {container_name} start timestamp changed during readiness wait "
                    f"({initial_started_at} -> {state['started_at']})"
                )
                print_container_logs_hint(container_name)
                stop_failed_startup_container(container_name)
                return False
            
            # Check logs for startup completion message
            result = subprocess.run(
                ['docker', 'logs', '--since', log_since, container_name],
                capture_output=True,
                text=True
            )
            
            log_output = result.stdout + result.stderr

            # Fail fast on known GPT-OSS harmony vocab error to avoid long pointless waits.
            if 'openai_harmony.HarmonyError: error downloading or loading vocab file' in log_output:
                print(f"\n✗ openai_harmony vocab download/load failed during startup")
                print("    Verify network/TLS access and persistent tiktoken cache availability.")
                print("    If needed, pre-seed cache online and then retry with --force-reload.")
                print_container_logs_hint(container_name)
                stop_failed_startup_container(container_name)
                return False

            if 'Application startup complete' in log_output:
                elapsed = int(time.time() - start_time)
                print(f" ready in {elapsed}s ✓")
                return True
            
            print('.', end='', flush=True)
            time.sleep(check_interval)
            
        except subprocess.CalledProcessError as e:
            print(f"\n✗ Error checking container: {e}")
            print_container_logs_hint(container_name)
            stop_failed_startup_container(container_name)
            return False
    
    if timeout is not None:
        print(f"\n✗ Timeout after {timeout}s")
        print_container_logs_hint(container_name)
        stop_failed_startup_container(container_name)
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


def start_existing_container(container_name: str) -> bool:
    """
    Start an existing Docker container.

    Args:
        container_name: Name of the container to start

    Returns:
        True if container started successfully, False otherwise
    """
    print(f"Starting existing container {container_name}...")
    try:
        subprocess.run(['docker', 'start', container_name], check=True, capture_output=True, text=True)
        print(f"✓ {container_name} started")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error starting {container_name}: {e}", file=sys.stderr)
        if e.stderr:
            print(f"  {e.stderr}", file=sys.stderr)
        print_container_logs_hint(container_name)
        return False


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
    cpuset_cpus = flat_params.get('cpuset-cpus', None)  # List of CPU IDs or string
    shm_size = flat_params.get('shm-size', None)  # Shared memory size (e.g., '8g', '512m')
    vllm_image = os.environ.get('VLLM_IMAGE', 'vllm/vllm-openai:v0.22.0-cu129')
    vllm_platform = os.environ.get('VLLM_PLATFORM')
    def _resolve_host_path(path_value: str) -> str:
        # Support values like ${HOME}/... and ~/... coming from .env.
        return os.path.expanduser(os.path.expandvars(path_value))

    tiktoken_cache_host = _resolve_host_path(
        os.environ.get('TIKTOKEN_RS_CACHE_DIR', '~/.cache/tiktoken-rs-cache')
    )
    huggingface_cache_host = _resolve_host_path(
        os.environ.get('HUGGINGFACE_CACHE', '~/.cache/huggingface')
    )

    # Keep openai_harmony/tiktoken artifacts on host so first online boot seeds offline runs.
    os.makedirs(tiktoken_cache_host, exist_ok=True)
    
    # Build docker run command
    cmd = [
        'docker', 'run',
        '-d',  # Detached mode
        '--name', container_name,
        '--runtime', 'nvidia',
        '--restart', 'unless-stopped',
    ]
    
    # Configure GPU access
    if gpus is None:
        # Use all available GPUs
        cmd.extend(['--gpus', 'all'])
    elif isinstance(gpus, list) and len(gpus) > 0:
        # Use specific GPU(s)
        gpu_list = ','.join(str(gpu) for gpu in gpus)
        cmd.extend(['--gpus', f'device={gpu_list}'])
    else:
        # Fallback to all GPUs
        cmd.extend(['--gpus', 'all'])
    
    # Configure CPU pinning
    if cpuset_cpus is not None:
        if isinstance(cpuset_cpus, list) and len(cpuset_cpus) > 0:
            # Convert list to comma-separated string
            cpu_list = ','.join(str(cpu) for cpu in cpuset_cpus)
            cmd.extend(['--cpuset-cpus', cpu_list])
        elif isinstance(cpuset_cpus, str):
            # Use string format directly (e.g., "0-5" or "0,1,2,3,4,5")
            cmd.extend(['--cpuset-cpus', cpuset_cpus])
    
    # Configure shared memory size
    if shm_size is not None:
        cmd.extend(['--shm-size', str(shm_size)])
    
    # Continue with rest of docker run options
    cmd.extend([
        '--ipc', 'host',
        '--network', network,
        '--label', 'inference.managed=true',
        '--label', f'inference.model={model_name}',
        '-v', f'{huggingface_cache_host}:/root/.cache/huggingface',
        '-v', f'{tiktoken_cache_host}:/tmp/tiktoken-rs-cache',
        '-e', 'TIKTOKEN_RS_CACHE_DIR=/tmp/tiktoken-rs-cache',
    ])
    
    # Add HF_TOKEN if available
    if 'HF_TOKEN' in os.environ:
        cmd.extend(['-e', f'HF_TOKEN={os.environ["HF_TOKEN"]}'])

    # In offline (air-gapped) environments, instruct HuggingFace libraries to
    # use only the local model cache and never attempt network requests.
    # This prevents vLLM's is_mistral_model_repo() / hf_list_repo_files()
    # from crashing with DNS resolution errors during model loading.
    offline = is_offline()
    if offline:
        print(f"  [offline] Injecting HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1 into {container_name}")
        cmd.extend([
            '-e', 'HF_HUB_OFFLINE=1',
            '-e', 'TRANSFORMERS_OFFLINE=1',
        ])

    if vllm_platform:
        cmd.extend(['--platform', vllm_platform])

    # Pull explicitly so users see progress before the detached container exists.
    # Skip when offline – the image must already be present locally.
    if offline:
        print(f"  [offline] Skipping image pull – using local image: {vllm_image}")
    else:
        pull_cmd = ['docker', 'pull']
        if vllm_platform:
            pull_cmd.extend(['--platform', vllm_platform])
        pull_cmd.append(vllm_image)
        print(f"Pulling image: {_redact_command(pull_cmd)}")
        try:
            subprocess.run(pull_cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error pulling vLLM image {vllm_image}: {e}", file=sys.stderr)
            return False
    
    # Add image and vllm command
    cmd.extend([
        vllm_image,
        model_id,
        '--port', str(port),
        '--host', '0.0.0.0'
    ])
    
    # Add all other parameters as vLLM flags
    for key, value in flat_params.items():
        if key not in ['model', 'gpus', 'cpuset-cpus', 'shm-size', 'additional_flags']:
            # Convert underscores to hyphens for command-line flags
            flag_name = key.replace('_', '-')
            cmd.extend([f'--{flag_name}', str(value)])
    
    # Add additional flags (for flags without values like --trust-remote-code)
    if 'additional_flags' in flat_params:
        additional_flags = flat_params['additional_flags']
        if isinstance(additional_flags, list):
            cmd.extend(additional_flags)
    
    print(f"Running: {_redact_command(cmd)}")
    
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
