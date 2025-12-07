<p align="center">
  <img src="https://raw.githubusercontent.com/likhonsheikhdev/docker-model-runner/main/assets/logo.svg" width="200" alt="Docker Model Runner Logo">
</p>

<h1 align="center">Docker Model Runner</h1>

<p align="center">
  <strong>Self-hosted Anthropic API Compatible Inference Server</strong>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#claude-code-integration">Claude Code</a> •
  <a href="#api-reference">API Reference</a> •
  <a href="#deployment">Deployment</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Anthropic_API-Compatible-blueviolet?style=for-the-badge" alt="Anthropic Compatible">
  <img src="https://img.shields.io/badge/OpenAI_API-Compatible-green?style=for-the-badge" alt="OpenAI Compatible">
  <img src="https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker" alt="Docker Ready">
  <img src="https://img.shields.io/badge/HuggingFace-Spaces-yellow?style=for-the-badge&logo=huggingface" alt="HuggingFace">
</p>

<p align="center">
  <img src="https://img.shields.io/github/license/likhonsheikhdev/docker-model-runner?style=flat-square" alt="License">
  <img src="https://img.shields.io/github/stars/likhonsheikhdev/docker-model-runner?style=flat-square" alt="Stars">
  <img src="https://img.shields.io/badge/python-3.11+-blue?style=flat-square" alt="Python">
</p>

---

## Overview

**Docker Model Runner** is a production-ready, self-hosted API server that provides full compatibility with the **Anthropic Messages API**. It enables you to run local language models while maintaining compatibility with tools like **Claude Code**, **Cursor**, and other AI-powered development tools.

### Why Docker Model Runner?

- **Drop-in Replacement**: Works seamlessly with any tool expecting Anthropic's API
- **Claude Code Compatible**: Use with Claude Code CLI for AI-assisted development
- **Interleaved Thinking**: Full support for extended thinking with streaming
- **Multi-API Support**: Both Anthropic and OpenAI API formats supported
- **CPU Optimized**: Runs efficiently on CPU-only hardware (2 vCPU, 16GB RAM)
- **Zero Configuration**: Deploy to HuggingFace Spaces with one click

---

## Features

| Feature | Status |
|---------|--------|
| Anthropic Messages API (`/v1/messages`) | ✅ Full Support |
| OpenAI Chat Completions (`/v1/chat/completions`) | ✅ Full Support |
| Streaming (SSE) | ✅ Full Support |
| Interleaved Thinking | ✅ Full Support |
| Tool/Function Calling | ✅ Full Support |
| Multi-turn Conversations | ✅ Full Support |
| System Prompts | ✅ Full Support |
| CORS (Agentic Tools) | ✅ Enabled |

---

## Quick Start

### Option 1: Use Hosted Version

```bash
# Set environment variables
export ANTHROPIC_BASE_URL=https://likhonsheikhdev-docker-model-runner.hf.space
export ANTHROPIC_API_KEY=your-api-key

# Use with Anthropic SDK
pip install anthropic
```

```python
import anthropic

client = anthropic.Anthropic()

message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}]
)

print(message.content[0].text)
```

### Option 2: Self-Host with Docker

```bash
# Clone the repository
git clone https://github.com/likhonsheikhdev/docker-model-runner.git
cd docker-model-runner

# Build and run
docker build -t model-runner .
docker run -p 7860:7860 model-runner
```

### Option 3: Deploy to HuggingFace Spaces

1. Fork this repository
2. Create a new Space on HuggingFace with Docker SDK
3. Upload the files
4. Your API is ready at `https://your-username-space-name.hf.space`

---

## Claude Code Integration

Use Docker Model Runner as your Claude Code backend:

### 1. Install Claude Code

```bash
# NPM
npm install -g @anthropic-ai/claude-code

# Homebrew (macOS)
brew install --cask claude-code

# Direct install
curl -fsSL https://claude.ai/install.sh | bash
```

### 2. Configure Settings

Create `~/.claude/settings.json`:

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "https://likhonsheikhdev-docker-model-runner.hf.space",
    "ANTHROPIC_API_KEY": "your-api-key",
    "API_TIMEOUT_MS": "300000"
  },
  "model": "claude-sonnet-4-20250514",
  "smallFastModel": "claude-sonnet-4-20250514"
}
```

### 3. Run Claude Code

```bash
claude
```

---

## Interleaved Thinking

Enable reasoning chains interleaved with responses:

```python
import anthropic

client = anthropic.Anthropic(
    base_url="https://likhonsheikhdev-docker-model-runner.hf.space"
)

message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=2048,
    thinking={
        "type": "enabled",
        "budget_tokens": 500
    },
    messages=[{"role": "user", "content": "Solve: What is 15% of 240?"}]
)

for block in message.content:
    if block.type == "thinking":
        print(f"💭 Thinking: {block.thinking}")
    elif block.type == "text":
        print(f"📝 Response: {block.text}")
```

### Streaming with Thinking

```python
with client.messages.stream(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    thinking={"type": "enabled", "budget_tokens": 200},
    messages=[{"role": "user", "content": "Explain quantum entanglement"}]
) as stream:
    for event in stream:
        if hasattr(event, 'type'):
            if event.type == 'content_block_delta':
                if hasattr(event.delta, 'thinking'):
                    print(event.delta.thinking, end="", flush=True)
                elif hasattr(event.delta, 'text'):
                    print(event.delta.text, end="", flush=True)
```

---

## API Reference

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/messages` | POST | Anthropic Messages API |
| `/anthropic/v1/messages` | POST | Alternative base path |
| `/api/v1/messages` | POST | Alternative base path |
| `/v1/chat/completions` | POST | OpenAI Chat API |
| `/v1/models` | GET | List available models |
| `/health` | GET | Health check |
| `/info` | GET | API information |

### Request Format

```bash
curl -X POST https://your-server/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: your-key" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "claude-sonnet-4-20250514",
    "max_tokens": 1024,
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

### Response Format

```json
{
  "id": "msg_abc123",
  "type": "message",
  "role": "assistant",
  "content": [
    {"type": "text", "text": "Hello! How can I help you today?"}
  ],
  "model": "claude-sonnet-4-20250514",
  "stop_reason": "end_turn",
  "usage": {
    "input_tokens": 10,
    "output_tokens": 25
  }
}
```

### Supported Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | string | Model identifier |
| `messages` | array | Conversation messages |
| `max_tokens` | integer | Maximum response tokens |
| `temperature` | float | Sampling temperature (0.0-1.0) |
| `top_p` | float | Nucleus sampling (0.0-1.0) |
| `stream` | boolean | Enable streaming |
| `system` | string | System prompt |
| `thinking` | object | Enable interleaved thinking |
| `tools` | array | Available tools |
| `tool_choice` | object | Tool selection preference |

---

## Deployment

### HuggingFace Spaces (Recommended)

```yaml
# README.md frontmatter
---
title: Docker Model Runner
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
suggested_hardware: cpu-basic
---
```

### Docker Compose

```yaml
version: '3.8'
services:
  model-runner:
    build: .
    ports:
      - "7860:7860"
    environment:
      - OMP_NUM_THREADS=2
      - MKL_NUM_THREADS=2
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 16G
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-runner
spec:
  replicas: 1
  template:
    spec:
      containers:
      - name: model-runner
        image: your-registry/model-runner:latest
        ports:
        - containerPort: 7860
        resources:
          limits:
            cpu: "2"
            memory: "16Gi"
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Model Runner                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │   /v1/       │  │  /anthropic/ │  │  /api/v1/    │       │
│  │   messages   │  │  v1/messages │  │  messages    │       │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘       │
│         │                 │                  │               │
│         └─────────────────┼──────────────────┘               │
│                           ▼                                  │
│                 ┌──────────────────┐                         │
│                 │  Request Handler │                         │
│                 │  (FastAPI)       │                         │
│                 └────────┬─────────┘                         │
│                          │                                   │
│         ┌────────────────┼────────────────┐                  │
│         ▼                ▼                ▼                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │   Thinking  │  │    Text     │  │    Tool     │          │
│  │  Generator  │  │  Generator  │  │   Handler   │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
│                          │                                   │
│                          ▼                                   │
│                 ┌──────────────────┐                         │
│                 │  Transformers    │                         │
│                 │  (CPU Optimized) │                         │
│                 └──────────────────┘                         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GENERATOR_MODEL` | `distilgpt2` | Text generation model |
| `MODEL_NAME` | `MiniMax-M2` | Display model name |
| `OMP_NUM_THREADS` | `2` | OpenMP threads |
| `MKL_NUM_THREADS` | `2` | MKL threads |

### Customizing Models

Edit `main.py` to use different models:

```python
GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "gpt2-medium")
```

---

## Performance

| Metric | Value |
|--------|-------|
| Cold Start | ~30s |
| Inference Latency | 50-200ms/token |
| Memory Usage | ~4GB |
| Concurrent Requests | 5-10 |

---

## Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) first.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- [HuggingFace](https://huggingface.co) for Spaces and Transformers
- [Anthropic](https://anthropic.com) for the API specification
- [FastAPI](https://fastapi.tiangolo.com) for the web framework

---

<p align="center">
  <strong>⭐ Star this repo if you find it useful!</strong>
</p>

<p align="center">
  Made with ❤️ by <a href="https://github.com/likhonsheikhdev">Likhon Sheikh</a>
</p>
