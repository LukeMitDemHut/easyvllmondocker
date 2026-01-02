"""
Command-line interface module for the serve application.
"""

import argparse
import sys
from typing import Optional, List, Tuple

from . import container_manager
from . import litellm_integration
from . import prometheus_integration
from . import docker_ops
from . import config


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description='Manage vLLM inference containers',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  serve                    Serve all models from config
  serve model1 model2      Serve specific models
  serve stop               Stop all running models
  serve stop model1        Stop a specific model
  serve down               Stop and remove all models
  serve down model1        Stop and remove a specific model
        """
    )
    
    parser.add_argument(
        'command',
        nargs='?',
        default='serve',
        help='Command to execute: serve (default), stop, or down'
    )
    
    parser.add_argument(
        'models',
        nargs='*',
        help='Specific model(s) to operate on (optional)'
    )
    
    parser.add_argument(
        '--force-reload',
        action='store_true',
        help='Force reload: stop, remove and restart existing containers even if they exist'
    )
    
    return parser.parse_args()


def resolve_command_and_models(args: argparse.Namespace) -> Tuple[str, Optional[List[str]]]:
    """
    Resolve the command and model names from parsed arguments.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Tuple of (command, model_names) where model_names can be None
    """
    # Parse command and models
    if args.command in ['stop', 'down']:
        command = args.command
        model_names = args.models if args.models else None
    elif args.command == 'serve':
        command = 'serve'
        model_names = args.models if args.models else None
    else:
        # First argument might be a model name, not a command
        command = 'serve'
        model_names = [args.command] + args.models if args.models else [args.command]
    
    return command, model_names


def execute_stop_command(model_names: Optional[List[str]]) -> None:
    """
    Execute the stop command.
    
    Args:
        model_names: List of model names to stop, or None for all models
    """
    container_manager.stop_models(model_names)


def execute_down_command(model_names: Optional[List[str]]) -> None:
    """
    Execute the down command.
    
    Args:
        model_names: List of model names to remove, or None for all models
    """
    container_manager.down_models(model_names)


def execute_serve_command(model_names: Optional[List[str]], force_reload: bool) -> None:
    """
    Execute the serve command.
    
    Args:
        model_names: List of model names to serve, or None for all models
        force_reload: Whether to force reload existing containers
    """
    # Load configuration to get model configs
    try:
        models = config.load_models_config()
    except config.ConfigurationError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    all_model_configs = config.get_all_model_configs(models)
    
    # Determine which models to serve
    if model_names:
        try:
            model_configs = config.validate_model_names(model_names, all_model_configs)
        except config.ConfigurationError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        model_configs = all_model_configs
    
    # Serve the models
    container_manager.serve_models(model_names, force_reload)
    
    # Update Prometheus configuration
    prometheus_config = prometheus_integration.get_prometheus_targets(model_configs)
    prometheus_integration.update_prometheus_config(prometheus_config)


def update_litellm_gateway() -> None:
    """
    Update the LiteLLM gateway configuration.
    """
    print("\nUpdating LiteLLM gateway to apply new configuration...")
    try:
        project_root = config.get_project_root()
        docker_ops.run_docker_compose(['up', '-d', 'litellm'], str(project_root))

        litellm_integration.update_litellm()

        print("✓ LiteLLM gateway started (with dependencies)")
    except docker_ops.DockerError as e:
        print(f"Error managing LiteLLM gateway: {e}", file=sys.stderr)
    except Exception as e:
        print(f"Error managing LiteLLM gateway: {e}", file=sys.stderr)

def start_grafana() -> None:
    """
    Start the Grafana service using Docker Compose.
    """
    print("\nStarting Grafana service...")
    try:
        project_root = config.get_project_root()
        docker_ops.run_docker_compose(['up', '-d', 'grafana'], str(project_root))
        print("✓ Grafana service running")
    except docker_ops.DockerError as e:
        print(f"Error starting Grafana service: {e}", file=sys.stderr)
    except Exception as e:
        print(f"Error starting Grafana service: {e}", file=sys.stderr)


def main() -> None:
    """
    Main entry point for the CLI application.
    """
    args = parse_arguments()
    command, model_names = resolve_command_and_models(args)
    
    # Execute command
    if command == 'stop':
        execute_stop_command(model_names)
    elif command == 'down':
        execute_down_command(model_names)
    else:  # serve
        execute_serve_command(model_names, args.force_reload)
        start_grafana()
    
    # Update LiteLLM gateway after any command
    update_litellm_gateway()
