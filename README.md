# EasyVllmOnDocker

**An easy way to set up a vLLM-based inference server on a single machine**, complete with:

- Multi-model configuration and deployment
- Unified API gateway with access control (LiteLLM)
- Full monitoring stack (Prometheus + Grafana)
- Simple YAML configuration and CLI management

Perfect for teams that need a **production-grade inference setup without the complexity** of Kubernetes or multi-node orchestration.

## 🎯 What This Is

This project provides a **batteries-included, single-machine LLM inference solution** that gets you from zero to production in minutes:

- **Single-machine deployment**: All components run on one GPU server via Docker Compose
- **vLLM-powered**: High-performance inference engine optimized for throughput
- **Multi-model support**: Run multiple models simultaneously with independent GPU allocation
- **Production features**: API gateway with authentication, request routing, and comprehensive monitoring
- **Simple management**: YAML config + CLI tool for all operations
- **Pre-configured monitoring**: Grafana dashboards for GPU, system, and inference metrics

**Use this if**: You have a powerful GPU server and want a simple, complete inference stack without DevOps overhead.

**Don't use this if**: You need multi-node, HA, auto-scaling, or have existing Kubernetes infrastructure.

## 📋 Prerequisites

### Hardware

- NVIDIA GPU(s) with sufficient VRAM (H200, H100, A100, or similar)
- NVIDIA drivers and Fabric Manager installed ([Cuda Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/))

### Software

- Docker Engine 20.10+ ([Installation Guide](https://docs.docker.com/engine/install/))
- Docker Compose 2.0+ ([Installation Guide](https://docs.docker.com/compose/install/))
- NVIDIA Container Toolkit ([Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html))
  - **Important**: After installing NVIDIA Container Toolkit, configure and restart Docker:
    ```bash
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    ```
- Python 3.8+ with pip and venv (for CLI tool)

### Accounts

- HuggingFace account with API token (for model downloads)

## ⚡ Quick Start

### 1. Clone and Install Dependencies

```bash
git clone https://github.com/LukeMitDemHut/easyvllmondocker.git
cd easyvllmondocker

# Install Python dependencies for CLI tool
# On Ubuntu 24.04+ and similar systems, use a virtual environment:
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt

# OR install system-wide (not recommended):
# pip3 install -r requirements.txt --break-system-packages

# Create environment configuration
cp .env.example .env
```

### 2. Configure Environment Variables

Edit `.env` with your credentials:

```bash
# Required: HuggingFace token for model downloads
HF_TOKEN=hf_your_token_here

# Required: Strong passwords and keys
POSTGRES_PASSWORD=your_secure_password
LITELLM_MASTER_KEY=sk-your_secure_key
LITELLM_SALT_KEY=your_random_salt
GRAFANA_ADMIN_PASSWORD=your_secure_password
```

See [Environment Variables Reference](#environment-variables-reference) for all options.

### 3. Configure Models

Edit `model_config/models.yaml`:

```yaml
models:
  - qwen-vl-4b-instruct:
      model: Qwen/Qwen3-VL-4B-Instruct
      gpus: [0]
      tensor-parallel-size: 1
      gpu-memory-utilization: 0.90
```

See [Model Configuration Reference](#model-configuration-reference) for all parameters.

### 4. Start Services

```bash
# Start infrastructure (gateway, database, monitoring)
docker compose up -d

# Deploy models
./serve
```

### 5. Access Services

- **API Gateway**: http://localhost:8000/ui
- **Grafana Dashboards**: http://localhost:3000 (admin/your-password)
- **Prometheus**: http://localhost:9090

## 📖 Usage

### API Usage

The server exposes an OpenAI-compatible API:

```python
import openai

client = openai.OpenAI(
    api_key="your-litellm-master-key",
    base_url="http://localhost:8000"
)

response = client.chat.completions.create(
    model="qwen-vl-4b-instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
)

print(response.choices[0].message.content)
```

Using cURL:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-litellm-master-key" \
  -d '{
    "model": "qwen-vl-4b-instruct",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### CLI Commands

```bash
# Deploy all models
./serve

# Deploy specific models
./serve model1 model2

# Stop models (keeps containers)
./serve stop
./serve stop model1

# Remove models (deletes containers)
./serve down
./serve down model1

# Force reload existing containers
./serve --force-reload
```

See [CLI Reference](#cli-reference) for detailed usage.

---

## 🔧 Configuration Reference

### Environment Variables Reference

#### Required Variables

| Variable                 | Description                               | Example         |
| ------------------------ | ----------------------------------------- | --------------- |
| `HF_TOKEN`               | HuggingFace API token for model downloads | `hf_...`        |
| `POSTGRES_PASSWORD`      | PostgreSQL database password              | Strong password |
| `LITELLM_MASTER_KEY`     | LiteLLM gateway API key                   | `sk-...`        |
| `LITELLM_SALT_KEY`       | LiteLLM encryption salt                   | Random string   |
| `GRAFANA_ADMIN_PASSWORD` | Grafana admin password                    | Strong password |

#### Optional Variables

| Variable            | Description           | Default                |
| ------------------- | --------------------- | ---------------------- |
| `GATEWAY_PORT`      | LiteLLM gateway port  | `8000`                 |
| `HUGGINGFACE_CACHE` | Model cache directory | `~/.cache/huggingface` |

### Model Configuration Reference

Models are defined in `model_config/models.yaml`.

#### File Structure

```yaml
models:
  - model-identifier:
      model: "huggingface/model-path" # Required
      gpus: [0, 1] # Optional: specific GPUs
      tensor-parallel-size: 2 # Optional: GPUs for sharding
      gpu-memory-utilization: 0.90 # Optional: memory fraction
      # Additional vLLM parameters...
```

#### Required Parameters

| Parameter | Description                       | Example                     |
| --------- | --------------------------------- | --------------------------- |
| `model`   | HuggingFace model repository path | `Qwen/Qwen3-VL-4B-Instruct` |

#### Common Parameters

| Parameter                | Description                          | Default  | Notes                              |
| ------------------------ | ------------------------------------ | -------- | ---------------------------------- |
| `gpus`                   | List of GPU IDs to use               | All GPUs | `[0]` or `[0, 1]`                  |
| `tensor-parallel-size`   | Number of GPUs to shard model across | `1`      | Must match GPU count               |
| `gpu-memory-utilization` | Fraction of GPU memory to use        | `0.9`    | Reduce for multiple models         |
| `max-model-len`          | Maximum sequence length              | Auto     | Override context window            |
| `dtype`                  | Data type for model weights          | `auto`   | `float16`, `bfloat16`, `float32`   |
| `quantization`           | Quantization method                  | `None`   | `awq`, `gptq`, `squeezellm`, `fp8` |
| `max-num-seqs`           | Maximum sequences per iteration      | `256`    | Reduce for lower memory            |
| `enable-prefix-caching`  | Enable prefix caching                | `true`   | Improves repeated prompts          |

All vLLM parameters supported. Use underscores (converted to hyphens automatically).

#### Multi-Model GPU Strategies

**Dedicated GPUs (Recommended):**

```yaml
models:
  - model-a:
      model: org/model-a
      gpus: [0]
      tensor-parallel-size: 1
      gpu-memory-utilization: 0.90
  - model-b:
      model: org/model-b
      gpus: [1]
      tensor-parallel-size: 1
      gpu-memory-utilization: 0.90
```

**Shared GPUs:**

```yaml
models:
  - model-a:
      model: org/model-a
      tensor-parallel-size: 2 # Use both GPUs
      gpu-memory-utilization: 0.25 # 25% of each
  - model-b:
      model: org/model-b
      tensor-parallel-size: 2
      gpu-memory-utilization: 0.50 # 50% of each
```

**Best Practices:**

- Use dedicated GPUs when possible to avoid memory conflicts
- Match `tensor-parallel-size` to GPU count in `gpus` list
- For shared GPUs, ensure sum of `gpu-memory-utilization` < 0.9 per GPU
- Monitor with `nvidia-smi`

---

## CLI Reference

### Commands

#### `./serve` (default)

Deploy models from configuration:

- Creates Docker containers for each model
- Skips already running models
- Registers models with LiteLLM gateway
- Updates Prometheus monitoring

**Examples:**

```bash
./serve                    # Deploy all models
./serve model1 model2      # Deploy specific models
./serve --force-reload     # Force restart existing
```

#### `./serve stop`

Stop running models without removing containers:

- Can restart later with `docker start`
- Preserves container state

**Examples:**

```bash
./serve stop              # Stop all models
./serve stop model1       # Stop specific model
```

#### `./serve down`

Stop and remove model containers:

- Completely removes containers
- Frees GPU memory

**Examples:**

```bash
./serve down              # Remove all models
./serve down model1       # Remove specific model
```

### How It Works

1. **Reads configuration** from `model_config/models.yaml`
2. **Creates containers** named `inference-<model-identifier>`
3. **Waits for startup** (monitors logs for "Application startup complete")
4. **Registers with LiteLLM** using retry logic with exponential backoff
5. **Updates Prometheus** targets for monitoring

### Custom Models in LiteLLM

Add external API models (OpenAI, Anthropic, etc.) that won't be managed by the serve script:

```bash
curl -X POST -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "gpt-4",
    "litellm_params": {
      "model": "gpt-4",
      "api_key": "sk-your-openai-key"
    }
  }' http://localhost:8000/model/new
```

The serve script only manages models with `organization: inference-vllm` tag.

---

## 🔌 Extending Configurations

### Extending vLLM Configuration

All vLLM server arguments can be added to model configurations in `model_config/models.yaml`:

```yaml
models:
  - advanced-model:
      model: org/model-name
      tensor-parallel-size: 2
      pipeline-parallel-size: 1
      max-num-batched-tokens: 8192
      max-num-seqs: 128
      enable-chunked-prefill: true
      disable-log-stats: false
      trust-remote-code: true
      # Any vLLM parameter works here
```

See [vLLM documentation](https://docs.vllm.ai/) for all available parameters.

### Extending LiteLLM Configuration

Edit `model_config/litellm_config.yaml` for gateway-level settings:

```yaml
litellm_settings:
  callbacks: ["prometheus"] # Monitoring integrations
  success_callback: []
  failure_callback: []

  # Add custom settings
  drop_params: true
  add_function_to_prompt: false

  # Rate limiting
  rpm: 60
  tpm: 100000
```

Models are registered dynamically via API by the serve script.
You will want to configure LiteLLM with different user groups, API-Keys and rate limits depending on your use case under the litellm-address eg. http://localhost:8000 .

See [LiteLLM documentation](https://docs.litellm.ai/) for all configuration options.

### Extending Prometheus Configuration

Edit `prometheus_config/prometheus.yml` to add custom scrape targets:

```yaml
global:
  scrape_interval: 5s
  evaluation_interval: 5s

scrape_configs:
  # Existing targets...

  # Add custom scrape target
  - job_name: "my-custom-service"
    static_configs:
      - targets: ["my-service:9090"]
    metric_relabel_configs:
      - source_labels: [__name__]
        regex: "custom_.*"
        action: keep
```

vLLM targets are managed automatically in `/etc/prometheus/vllm_targets.json`.

### Extending Grafana Configuration

#### Add Custom Datasources

Create files in `grafana_config/datasources/`:

```yaml
# grafana_config/datasources/custom.yaml
apiVersion: 1

datasources:
  - name: CustomDB
    type: postgres
    url: postgres:5432
    database: mydb
    user: admin
    secureJsonData:
      password: ${POSTGRES_PASSWORD}
```

#### Add Custom Dashboards

1. Create dashboard JSON in `grafana_config/dashboards/`
2. Register in `grafana_config/dashboards.yaml`:

```yaml
apiVersion: 1

providers:
  - name: "Default"
    folder: "General"
    type: file
    options:
      path: /etc/grafana/provisioning/dashboards
```

3. Restart Grafana: `docker-compose restart grafana`

---

## 📊 Monitoring

Access Grafana at http://localhost:3000 or your configured port.

### Available Dashboards

- **Home Dashboard**: System overview, GPU utilization, gateway metrics
- **System Dashboard**: CPU, memory, disk, network, GPU temperature
- **LiteLLM Dashboard**: Request rates, latency, success/error rates
- **vLLM Dashboard**: Model queues, KV cache, generation throughput

### Prometheus Targets

Access Prometheus at http://localhost:9090/targets to verify all services are up:

- LiteLLM gateway (port 4000)
- vLLM models (auto-configured)
- Node Exporter (system metrics)
- DCGM Exporter (GPU metrics)

---

## 🐛 Troubleshooting

### Models Not Starting

**Check container logs:**

```bash
docker logs inference-model-name
```

**Verify GPU availability:**

```bash
nvidia-smi
```

**Common issues:**

- Out of GPU memory: Reduce `gpu-memory-utilization` in models.yaml
- Model download failed: Check `HF_TOKEN` is set correctly
- Container exits immediately: Check logs for configuration errors
- **KV cache memory error** (`ValueError: To serve at least one request...`): The model's default context length is too large for available GPU memory. Add `max-model-len` to your model config to reduce context window:
  ```yaml
  models:
    - your-model:
        model: org/model-name
        max-model-len: 51216 # Use the suggested value from error message
        gpu-memory-utilization: 0.90
  ```

### LiteLLM Connection Issues

**Verify service is running:**

```bash
docker ps | grep litellm
docker logs litellm
```

**Test model registration:**

```bash
curl -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
     http://localhost:8000/model/info
```

**Common issues:**

- "LiteLLM not ready yet": Normal during startup, CLI will retry automatically
- "LITELLM_MASTER_KEY not found": Set in `.env` and restart services
- Models not registered: Check container logs and network connectivity

### Prometheus/Grafana Issues

**Restart monitoring stack:**

```bash
docker-compose restart prometheus grafana
```

**Verify targets are up:**

- Visit http://localhost:9090/targets
- All targets should show status "UP"

**Dashboard not showing data:**

- Check Prometheus is scraping targets successfully
- Verify datasource configuration in Grafana
- Check time range in dashboard (default: last 5 minutes)

### Performance Issues

**High GPU memory usage:**

- Reduce `gpu-memory-utilization` in models.yaml
- Use dedicated GPU assignment instead of shared

**Slow inference:**

- Check GPU utilization with `nvidia-smi`
- Increase `max-num-seqs` for higher throughput
- Enable `enable-prefix-caching` for repeated prompts
- Monitor KV cache usage in vLLM dashboard

**Network connectivity:**

```bash
# Check Docker network
docker network inspect inference_default

# Test container connectivity
docker exec inference-model-name curl localhost:8000/health
```

---

## 🙏 Acknowledgments

- [vLLM](https://github.com/vllm-project/vllm) - High-performance LLM inference
- [LiteLLM](https://github.com/BerriAI/litellm) - Unified LLM gateway
- [Prometheus](https://prometheus.io/) - Monitoring and alerting
- [Grafana](https://grafana.com/) - Metrics visualization

## ⚖️ License

This project is licensed under the [Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)](LICENSE) License.

**Note on Scope:** This license applies **only** to the original code, CLI tools (`./serve`), and configuration files within this repository. It does **not** apply to the third-party software this project orchestrates—including but not limited to **vLLM**, **LiteLLM**, **Prometheus**, and **Grafana**—which are governed by their own respective licenses.
