# Module Architecture

## Dependency Graph

```
serve (entry point)
  └─→ cli.py (main orchestration)
       ├─→ config.py (configuration management)
       ├─→ container_manager.py (high-level operations)
       │    ├─→ docker_ops.py (low-level Docker operations)
       │    └─→ config.py
       ├─→ litellm_integration.py (LiteLLM sync)
       │    ├─→ docker_ops.py
       │    └─→ config.py
       └─→ prometheus_integration.py (Prometheus config)
            ├─→ docker_ops.py
            └─→ config.py
```

## Module Responsibilities

### Core Modules

**docker_ops.py** (Foundation Layer)
- Container CRUD operations
- Network management
- Health checking
- Docker compose operations
- Docker exec operations

**config.py** (Foundation Layer)
- YAML configuration loading
- Model configuration parsing
- Path resolution
- Configuration validation

### Business Logic Modules

**container_manager.py** (Orchestration Layer)
- Implements serve/stop/down commands
- Manages container lifecycle
- Coordinates Docker operations
- Handles force-reload logic

**litellm_integration.py** (Integration Layer)
- LiteLLM API communication
- Model synchronization
- Add/remove model operations

**prometheus_integration.py** (Integration Layer)
- Prometheus configuration generation
- Container configuration updates
- Configuration reload

### Interface Module

**cli.py** (Presentation Layer)
- Argument parsing
- Command routing
- Workflow orchestration
- Integration of all subsystems

## Data Flow

### Serve Command Flow
```
1. cli.py → parse arguments
2. cli.py → config.py → load models.yaml
3. cli.py → container_manager.serve_models()
4. container_manager → docker_ops → start containers
5. cli.py → prometheus_integration → update Prometheus
6. cli.py → litellm_integration → sync LiteLLM
```

### Stop Command Flow
```
1. cli.py → parse arguments
2. cli.py → container_manager.stop_models()
3. container_manager → docker_ops → stop containers
4. cli.py → litellm_integration → sync LiteLLM
```

### Down Command Flow
```
1. cli.py → parse arguments
2. cli.py → container_manager.down_models()
3. container_manager → docker_ops → remove containers
4. cli.py → litellm_integration → sync LiteLLM
```
