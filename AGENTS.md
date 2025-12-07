# AGENTS.md - AI Agent Instructions

## Overview

This file provides instructions for AI coding agents (Claude Code, Cursor, Aider, etc.) working with this codebase.

---

## Project Structure

```
docker-model-runner/
├── main.py              # FastAPI application (core logic)
├── Dockerfile           # Docker build configuration
├── docker-compose.yml   # Docker Compose setup
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
├── AGENTS.md            # This file - AI agent instructions
├── LICENSE              # MIT License
├── CONTRIBUTING.md      # Contribution guidelines
├── github/
│   └── instructions.md  # GitHub workflow instructions
├── assets/
│   └── logo.svg         # Project logo
└── static/
    └── index.html       # Frontend page
```

---

## Key Components

### main.py

The core FastAPI application with these main sections:

1. **Models & Configuration** (Lines 1-50)
   - Environment variables
   - Model loading
   - Global state

2. **Pydantic Models** (Lines 70-180)
   - `AnthropicRequest`: Main request model
   - `AnthropicResponse`: Response model
   - Content blocks: `TextBlock`, `ThinkingBlock`, `ToolUseBlock`

3. **Helper Functions** (Lines 180-300)
   - `format_messages_to_prompt()`: Convert messages to prompt
   - `generate_text()`: Text generation
   - `generate_thinking()`: Thinking block generation
   - `generate_stream_with_thinking()`: Streaming generator

4. **API Endpoints** (Lines 400-530)
   - `POST /v1/messages`: Anthropic API
   - `POST /anthropic/v1/messages`: Alternative path
   - `POST /v1/chat/completions`: OpenAI API
   - `GET /v1/models`: List models
   - `GET /health`: Health check

---

## Common Tasks

### Adding a New Endpoint

```python
@app.post("/v1/new-endpoint")
async def new_endpoint(request: NewRequest):
    """Description of endpoint."""
    # Implementation
    return {"result": "success"}
```

### Adding a New Content Block Type

1. Create Pydantic model:
```python
class NewBlock(BaseModel):
    type: Literal["new_type"] = "new_type"
    content: str
```

2. Add to `ContentBlock` union:
```python
ContentBlock = Union[TextBlock, ThinkingBlock, NewBlock, ...]
```

3. Handle in response generation.

### Modifying Streaming Behavior

The streaming is handled in `generate_stream_with_thinking()`. Key events:
- `message_start`: Initial message
- `content_block_start`: New content block
- `content_block_delta`: Content updates
- `content_block_stop`: Block complete
- `message_delta`: Final usage stats
- `message_stop`: End of stream

---

## API Compatibility Notes

### Anthropic API Requirements

- Response must include `id`, `type`, `role`, `content`, `model`, `stop_reason`, `usage`
- Content is an array of blocks, not a string
- Streaming uses Server-Sent Events (SSE)
- `stop_reason` values: `end_turn`, `max_tokens`, `stop_sequence`, `tool_use`

### Thinking/Extended Thinking

When `thinking.type == "enabled"`:
1. Generate thinking content first
2. Include `ThinkingBlock` before `TextBlock`
3. In streaming, send thinking deltas before text deltas

---

## Testing Changes

```bash
# Quick API test
curl -X POST http://localhost:7860/v1/messages \
  -H "Content-Type: application/json" \
  -d '{"model":"test","max_tokens":50,"messages":[{"role":"user","content":"Hi"}]}'

# Test streaming
curl -X POST http://localhost:7860/v1/messages \
  -H "Content-Type: application/json" \
  -d '{"model":"test","max_tokens":50,"stream":true,"messages":[{"role":"user","content":"Hi"}]}'

# Test thinking
curl -X POST http://localhost:7860/v1/messages \
  -H "Content-Type: application/json" \
  -d '{"model":"test","max_tokens":100,"thinking":{"type":"enabled","budget_tokens":50},"messages":[{"role":"user","content":"What is 2+2?"}]}'
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GENERATOR_MODEL` | `distilgpt2` | HuggingFace model for text generation |
| `MODEL_NAME` | `MiniMax-M2` | Display name in responses |
| `OMP_NUM_THREADS` | `2` | OpenMP thread count |
| `MKL_NUM_THREADS` | `2` | MKL thread count |

---

## Important Patterns

### Error Handling

```python
try:
    # Operation
except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))
```

### Response Format

Always return proper Anthropic format:
```python
return AnthropicResponse(
    id=f"msg_{uuid.uuid4().hex[:24]}",
    content=[TextBlock(type="text", text="Response")],
    model=request.model,
    stop_reason="end_turn",
    usage=Usage(input_tokens=10, output_tokens=20)
)
```

---

## Do's and Don'ts

### Do
- Maintain Anthropic API compatibility
- Use proper type hints
- Handle both streaming and non-streaming
- Include proper error handling
- Test with Claude Code after changes

### Don't
- Break existing API endpoints
- Remove required response fields
- Change response format without compatibility layer
- Ignore the `thinking` parameter
- Forget to update documentation

---

## Contact

- **Author**: Likhon Sheikh
- **Telegram**: [@likhonsheikh](https://t.me/likhonsheikh)
- **GitHub**: [@likhonsheikhdev](https://github.com/likhonsheikhdev)

---

*This file helps AI agents understand and work with the Docker Model Runner codebase effectively.*
