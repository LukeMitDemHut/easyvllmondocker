# Serve App - Modular LLM Container Management

This is a refactored, modular Python application for managing vLLM inference containers. The application has been restructured from a monolithic script into maintainable, well-organized modules.

## Project Structure

```
scripts/
├── serve                          # Main entry point script
└── serve_app/                     # Application package
    ├── __init__.py               # Package initialization
    ├── cli.py                    # Command-line interface and main orchestration
    ├── config.py                 # Configuration loading and management
    ├── container_manager.py      # High-level container lifecycle management
    ├── docker_ops.py             # Low-level Docker operations
    ├── litellm_integration.py    # LiteLLM gateway synchronization
    └── prometheus_integration.py # Prometheus monitoring configuration
```

## Module Overview

### `cli.py` - Command-Line Interface
- Handles argument parsing
- Orchestrates commands (serve, stop, down)
- Manages the overall workflow
- Entry point for the application

### `config.py` - Configuration Management
- Loads and parses `models.yaml` configuration
- Validates model configurations
- Provides utility functions for configuration access

### `docker_ops.py` - Docker Operations
- Low-level Docker container operations (start, stop, remove)
- Container status checking and monitoring
- Network management
- Container health checking and readiness waiting
- Supports advanced configuration:
  - GPU selection via `gpus` parameter
  - CPU pinning via `cpuset-cpus` parameter
  - Additional CLI flags via `additional_flags` list
  - Automatic parameter conversion (underscores to hyphens)

### `container_manager.py` - Container Management
- High-level orchestration of container lifecycle
- Implements serve, stop, and down operations
- Manages multiple containers sequentially
- Handles force-reload logic

### `litellm_integration.py` - LiteLLM Integration
- Synchronizes models with LiteLLM gateway
- Adds/removes models from LiteLLM based on running containers
- Handles LiteLLM API communication

### `prometheus_integration.py` - Prometheus Integration
- Generates Prometheus target configurations
- Updates Prometheus with current model endpoints
- Reloads Prometheus configuration

## Usage

The refactored application maintains the exact same command-line interface:

```bash
# Serve all models from configuration
./serve

# Serve specific models
./serve model1 model2

# Stop all models
./serve stop

# Stop specific models
./serve stop model1

# Remove all models
./serve down

# Remove specific models
./serve down model1

# Force reload existing containers
./serve --force-reload
```

## Benefits of the Modular Structure

1. **Maintainability**: Each module has a single, clear responsibility
2. **Testability**: Modules can be tested independently
3. **Reusability**: Functions can be imported and reused
4. **Readability**: Smaller files are easier to understand
5. **Extensibility**: New features can be added without affecting existing code
6. **Error Handling**: Centralized error types per module

## Development

To modify functionality:

1. **Docker operations**: Edit `docker_ops.py`
2. **Configuration logic**: Edit `config.py`
3. **Container orchestration**: Edit `container_manager.py`
4. **LiteLLM sync**: Edit `litellm_integration.py`
5. **Prometheus updates**: Edit `prometheus_integration.py`
6. **CLI behavior**: Edit `cli.py`

## Dependencies

- Python 3.6+
- PyYAML
- Docker
- Docker Compose

All dependencies are standard library except for PyYAML, which is used for configuration parsing.
