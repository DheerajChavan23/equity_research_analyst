from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class ToolCall(BaseModel):
    id: str
    name: str
    arguments: Dict[str, Any]
    thought_signature: Optional[Any] = None

class Message(BaseModel):
    role: str  # "system", "user", "assistant", "tool"
    content: str
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None

class LLMRequest(BaseModel):
    messages: List[Message]
    temperature: float = 0.0
    response_format: Optional[str] = None  # e.g., "json"
    tools: Optional[List[Dict[str, Any]]] = None # Standard JSON Schema (OpenAI format)

class LLMResponse(BaseModel):
    content: str
    tool_calls: Optional[List[ToolCall]] = None
    input_tokens: int = 0
    output_tokens: int = 0
    model_name: str