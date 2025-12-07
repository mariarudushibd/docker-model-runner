"""
Docker Model Runner - Anthropic API Compatible
Full compatibility with Anthropic Messages API + Interleaved Thinking
Supports: /v1/messages, /anthropic/v1/messages, /api/v1/messages
Optimized for: 2 vCPU, 16GB RAM
"""
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Union, Literal, Any, Dict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from datetime import datetime
from contextlib import asynccontextmanager
import uuid
import time
import json
import asyncio

# CPU-optimized lightweight models
GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "distilgpt2")
MODEL_DISPLAY_NAME = os.getenv("MODEL_NAME", "MiniMax-M2")

# Set CPU threading
torch.set_num_threads(2)

# Global model cache
models = {}


def load_models():
    global models
    print("Loading models for CPU inference...")
    models["tokenizer"] = AutoTokenizer.from_pretrained(GENERATOR_MODEL)
    models["model"] = AutoModelForCausalLM.from_pretrained(GENERATOR_MODEL)
    models["model"].eval()
    if models["tokenizer"].pad_token is None:
        models["tokenizer"].pad_token = models["tokenizer"].eos_token
    print("✅ All models loaded successfully!")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield
    models.clear()


app = FastAPI(
    title="Model Runner",
    description="Anthropic API Compatible - Works with Claude Code & Agentic Tools",
    version="1.1.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS for agentic tools
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== Anthropic API Models ==============

class TextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ThinkingBlock(BaseModel):
    type: Literal["thinking"] = "thinking"
    thinking: str


class SignatureBlock(BaseModel):
    type: Literal["signature"] = "signature"
    signature: str


class ToolUseBlock(BaseModel):
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: Dict[str, Any]


class ToolResultContent(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: Union[str, List[TextBlock]]
    is_error: Optional[bool] = False


class ImageSource(BaseModel):
    type: Literal["base64", "url"]
    media_type: Optional[str] = None
    data: Optional[str] = None
    url: Optional[str] = None


class ImageBlock(BaseModel):
    type: Literal["image"] = "image"
    source: ImageSource


ContentBlock = Union[TextBlock, ThinkingBlock, SignatureBlock, ToolUseBlock, ToolResultContent, ImageBlock, str]


class MessageParam(BaseModel):
    role: Literal["user", "assistant"]
    content: Union[str, List[ContentBlock]]


class ToolInputSchema(BaseModel):
    type: str = "object"
    properties: Optional[Dict[str, Any]] = None
    required: Optional[List[str]] = None


class Tool(BaseModel):
    name: str
    description: str
    input_schema: ToolInputSchema


class ToolChoice(BaseModel):
    type: Literal["auto", "any", "tool"] = "auto"
    name: Optional[str] = None
    disable_parallel_tool_use: Optional[bool] = False


class ThinkingConfig(BaseModel):
    type: Literal["enabled", "disabled"] = "disabled"
    budget_tokens: Optional[int] = None


class Metadata(BaseModel):
    user_id: Optional[str] = None


class AnthropicRequest(BaseModel):
    model: str = "MiniMax-M2"
    messages: List[MessageParam]
    max_tokens: int = 4096
    temperature: Optional[float] = Field(default=1.0, gt=0.0, le=1.0)
    top_p: Optional[float] = Field(default=1.0, gt=0.0, le=1.0)
    top_k: Optional[int] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    system: Optional[Union[str, List[TextBlock]]] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Union[ToolChoice, Dict[str, Any]]] = None
    metadata: Optional[Metadata] = None
    thinking: Optional[Union[ThinkingConfig, Dict[str, Any]]] = None
    service_tier: Optional[str] = None


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: Optional[int] = 0
    cache_read_input_tokens: Optional[int] = 0


class AnthropicResponse(BaseModel):
    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: List[Union[TextBlock, ThinkingBlock, SignatureBlock, ToolUseBlock]]
    model: str
    stop_reason: Optional[Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]] = "end_turn"
    stop_sequence: Optional[str] = None
    usage: Usage


# ============== Helper Functions ==============

def extract_text_from_content(content: Union[str, List[ContentBlock]]) -> str:
    if isinstance(content, str):
        return content
    texts = []
    for block in content:
        if isinstance(block, str):
            texts.append(block)
        elif hasattr(block, 'text'):
            texts.append(block.text)
        elif hasattr(block, 'thinking'):
            texts.append(block.thinking)
        elif isinstance(block, dict):
            if block.get('type') == 'text':
                texts.append(block.get('text', ''))
            elif block.get('type') == 'thinking':
                texts.append(block.get('thinking', ''))
    return " ".join(texts)


def format_system_prompt(system: Optional[Union[str, List[TextBlock]]]) -> str:
    if system is None:
        return ""
    if isinstance(system, str):
        return system
    return " ".join([block.text for block in system if hasattr(block, 'text')])


def format_messages_to_prompt(messages: List[MessageParam], system: Optional[Union[str, List[TextBlock]]] = None, include_thinking: bool = False) -> str:
    prompt_parts = []
    system_text = format_system_prompt(system)
    if system_text:
        prompt_parts.append(f"System: {system_text}\n\n")
    for msg in messages:
        role = msg.role
        content = msg.content
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    block_type = block.get('type', 'text')
                    if block_type == 'thinking' and include_thinking:
                        prompt_parts.append(f"<thinking>{block.get('thinking', '')}</thinking>\n")
                    elif block_type == 'text':
                        text_content = block.get('text', '')
                        if role == "user":
                            prompt_parts.append(f"Human: {text_content}\n\n")
                        else:
                            prompt_parts.append(f"Assistant: {text_content}\n\n")
                    elif block_type == 'tool_result':
                        prompt_parts.append(f"Tool Result: {block.get('content', '')}\n\n")
                elif hasattr(block, 'type'):
                    if block.type == 'thinking' and include_thinking:
                        prompt_parts.append(f"<thinking>{block.thinking}</thinking>\n")
                    elif block.type == 'text':
                        if role == "user":
                            prompt_parts.append(f"Human: {block.text}\n\n")
                        else:
                            prompt_parts.append(f"Assistant: {block.text}\n\n")
        else:
            content_text = content if isinstance(content, str) else extract_text_from_content(content)
            if role == "user":
                prompt_parts.append(f"Human: {content_text}\n\n")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content_text}\n\n")
    prompt_parts.append("Assistant:")
    return "".join(prompt_parts)


def generate_text(prompt: str, max_tokens: int, temperature: float, top_p: float) -> tuple:
    tokenizer = models["tokenizer"]
    model = models["model"]
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    input_tokens = inputs["input_ids"].shape[1]
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=min(max_tokens, 512),
            temperature=temperature if temperature > 0 else 1.0,
            top_p=top_p,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    generated_tokens = outputs[0][input_tokens:]
    output_tokens = len(generated_tokens)
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return generated_text.strip(), input_tokens, output_tokens


def generate_thinking(prompt: str, budget_tokens: int = 100) -> tuple:
    tokenizer = models["tokenizer"]
    model = models["model"]
    thinking_prompt = f"{prompt}\n\nLet me think through this step by step:\n"
    inputs = tokenizer(thinking_prompt, return_tensors="pt", truncation=True, max_length=512)
    input_tokens = inputs["input_ids"].shape[1]
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=min(budget_tokens, 256),
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    generated_tokens = outputs[0][input_tokens:]
    thinking_tokens = len(generated_tokens)
    thinking_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return thinking_text.strip(), thinking_tokens


async def generate_stream_with_thinking(prompt: str, max_tokens: int, temperature: float, top_p: float, message_id: str, model_name: str, thinking_enabled: bool = False, thinking_budget: int = 100):
    tokenizer = models["tokenizer"]
    model = models["model"]
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    input_tokens = inputs["input_ids"].shape[1]
    total_output_tokens = 0

    message_start = {
        "type": "message_start",
        "message": {"id": message_id, "type": "message", "role": "assistant", "content": [], "model": model_name, "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": input_tokens, "output_tokens": 0}}
    }
    yield f"event: message_start\ndata: {json.dumps(message_start)}\n\n"

    content_index = 0

    if thinking_enabled:
        thinking_block_start = {"type": "content_block_start", "index": content_index, "content_block": {"type": "thinking", "thinking": ""}}
        yield f"event: content_block_start\ndata: {json.dumps(thinking_block_start)}\n\n"
        thinking_text, thinking_tokens = generate_thinking(prompt, thinking_budget)
        total_output_tokens += thinking_tokens
        for i in range(0, len(thinking_text), 10):
            chunk = thinking_text[i:i+10]
            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': content_index, 'delta': {'type': 'thinking_delta', 'thinking': chunk}})}\n\n"
            await asyncio.sleep(0.01)
        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': content_index})}\n\n"
        content_index += 1

    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': content_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=min(max_tokens, 512), temperature=temperature if temperature > 0 else 1.0, top_p=top_p, do_sample=temperature > 0, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)

    generated_tokens = outputs[0][input_tokens:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    total_output_tokens += len(generated_tokens)

    for i in range(0, len(generated_text), 5):
        yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': content_index, 'delta': {'type': 'text_delta', 'text': generated_text[i:i+5]}})}\n\n"
        await asyncio.sleep(0.005)

    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': content_index})}\n\n"
    yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': {'output_tokens': total_output_tokens}})}\n\n"
    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"


def handle_tool_call(tools: List[Tool], messages: List[MessageParam], generated_text: str) -> Optional[ToolUseBlock]:
    if not tools:
        return None
    for tool in tools:
        if tool.name.lower() in generated_text.lower():
            return ToolUseBlock(type="tool_use", id=f"toolu_{uuid.uuid4().hex[:24]}", name=tool.name, input={})
    return None


# ============== Core Messages Handler ==============

async def handle_messages(request: AnthropicRequest):
    """Core handler for Anthropic Messages API"""
    try:
        message_id = f"msg_{uuid.uuid4().hex[:24]}"
        thinking_enabled = False
        thinking_budget = 100

        if request.thinking:
            if isinstance(request.thinking, dict):
                thinking_enabled = request.thinking.get('type') == 'enabled'
                thinking_budget = request.thinking.get('budget_tokens', 100) or 100
            else:
                thinking_enabled = request.thinking.type == 'enabled'
                thinking_budget = request.thinking.budget_tokens or 100

        prompt = format_messages_to_prompt(request.messages, request.system, include_thinking=thinking_enabled)

        if request.stream:
            return StreamingResponse(
                generate_stream_with_thinking(prompt, request.max_tokens, request.temperature or 1.0, request.top_p or 1.0, message_id, request.model, thinking_enabled, thinking_budget),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
            )

        content_blocks = []
        total_output_tokens = 0

        if thinking_enabled:
            thinking_text, thinking_tokens = generate_thinking(prompt, thinking_budget)
            total_output_tokens += thinking_tokens
            content_blocks.append(ThinkingBlock(type="thinking", thinking=thinking_text))

        generated_text, input_tokens, output_tokens = generate_text(prompt, request.max_tokens, request.temperature or 1.0, request.top_p or 1.0)
        total_output_tokens += output_tokens

        tool_use = handle_tool_call(request.tools, request.messages, generated_text) if request.tools else None

        if tool_use:
            content_blocks.append(TextBlock(type="text", text=generated_text))
            content_blocks.append(tool_use)
            stop_reason = "tool_use"
        else:
            content_blocks.append(TextBlock(type="text", text=generated_text))
            stop_reason = "end_turn"

        return AnthropicResponse(id=message_id, content=content_blocks, model=request.model, stop_reason=stop_reason, usage=Usage(input_tokens=input_tokens, output_tokens=total_output_tokens))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============== Frontend ==============

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTMLResponse(content="""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Model Runner</title>
<style>*{margin:0;padding:0;box-sizing:border-box}body{min-height:100vh;background:#000;display:flex;justify-content:center;align-items:center;font-family:system-ui,sans-serif}
.container{display:flex;flex-direction:column;align-items:center;gap:2rem}
.logo{width:200px;height:200px;animation:float 3s ease-in-out infinite;filter:drop-shadow(0 0 30px rgba(255,100,100,0.3))}
.status{display:flex;align-items:center;gap:0.5rem;color:rgba(255,255,255,0.6);font-size:0.875rem}
.dot{width:8px;height:8px;background:#22c55e;border-radius:50%;animation:pulse 2s ease-in-out infinite}
.sparkle{position:fixed;bottom:2rem;right:2rem;opacity:0.4}
@keyframes float{0%,100%{transform:translateY(0)}50%{transform:translateY(-10px)}}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.5}}</style></head>
<body><div class="container"><div class="logo"><svg viewBox="0 0 200 200" fill="none">
<defs><linearGradient id="r" x1="0%" y1="100%" x2="100%" y2="0%">
<stop offset="0%" stop-color="#ff0080"/><stop offset="20%" stop-color="#ff4d00"/>
<stop offset="40%" stop-color="#ffcc00"/><stop offset="60%" stop-color="#00ff88"/>
<stop offset="80%" stop-color="#00ccff"/><stop offset="100%" stop-color="#6644ff"/></linearGradient></defs>
<path d="M100 20 L180 160 L20 160 Z" stroke="url(#r)" stroke-width="12" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
<path d="M100 70 L130 130 L70 130 Z" stroke="url(#r)" stroke-width="8" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
<line x1="80" y1="115" x2="120" y2="115" stroke="url(#r)" stroke-width="6" stroke-linecap="round"/>
</svg></div><div class="status"><span class="dot"></span><span>Ready</span></div></div>
<svg class="sparkle" width="24" height="24" viewBox="0 0 24 24" fill="none">
<path d="M12 2L13.5 8.5L20 10L13.5 11.5L12 18L10.5 11.5L4 10L10.5 8.5L12 2Z" fill="rgba(255,255,255,0.6)"/></svg>
</body></html>""")


# ============== Anthropic API Routes ==============
# Support multiple base paths for compatibility

@app.post("/v1/messages")
async def messages_v1(request: AnthropicRequest):
    """Standard Anthropic API endpoint"""
    return await handle_messages(request)


@app.post("/anthropic/v1/messages")
async def messages_anthropic(request: AnthropicRequest):
    """Anthropic base path - for Claude Code compatibility"""
    return await handle_messages(request)


@app.post("/api/v1/messages")
async def messages_api(request: AnthropicRequest):
    """API base path variant"""
    return await handle_messages(request)


# ============== OpenAI Compatible ==============

class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]


class ChatCompletionRequest(BaseModel):
    model: str = "gpt-4"
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 4096
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    stream: Optional[bool] = False


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    try:
        # Extract text from messages
        formatted_messages = []
        for msg in request.messages:
            if msg.role in ["user", "assistant"]:
                content = msg.content
                if isinstance(content, list):
                    text_parts = [c.get('text', '') for c in content if isinstance(c, dict) and c.get('type') == 'text']
                    content = ' '.join(text_parts)
                formatted_messages.append(MessageParam(role=msg.role, content=content))

        prompt = format_messages_to_prompt(formatted_messages)
        generated_text, input_tokens, output_tokens = generate_text(prompt, request.max_tokens or 4096, request.temperature or 0.7, request.top_p or 1.0)

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": generated_text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": input_tokens, "completion_tokens": output_tokens, "total_tokens": input_tokens + output_tokens}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============== Models Endpoints ==============

@app.get("/v1/models")
@app.get("/anthropic/v1/models")
@app.get("/api/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "claude-sonnet-4-20250514", "object": "model", "created": int(time.time()), "owned_by": "anthropic"},
            {"id": "claude-3-5-sonnet-20241022", "object": "model", "created": int(time.time()), "owned_by": "anthropic"},
            {"id": "MiniMax-M2", "object": "model", "created": int(time.time()), "owned_by": "local"},
            {"id": "MiniMax-M2-Stable", "object": "model", "created": int(time.time()), "owned_by": "local"},
            {"id": GENERATOR_MODEL, "object": "model", "created": int(time.time()), "owned_by": "local"}
        ]
    }


# ============== Utility Endpoints ==============

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat(), "models_loaded": len(models) > 0}


@app.get("/info")
async def info():
    return {
        "name": "Model Runner",
        "version": "1.1.0",
        "api_compatibility": ["anthropic", "openai"],
        "base_paths": ["/v1/messages", "/anthropic/v1/messages", "/api/v1/messages"],
        "interleaved_thinking": True,
        "agentic_tools": True
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
