import os
import json
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

from .schema import Message, ToolCall, LLMRequest, LLMResponse
from ops.telemetry import trace_llm_call

class LLMGateway:
    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None):
        self.provider = (provider or os.getenv("RESEARCH_LLM_PROVIDER", "openai")).lower()
        self.model = model or os.getenv("DEFAULT_MODEL", "gpt-4o")
        
        if self.provider == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif self.provider == "anthropic":
            from anthropic import Anthropic
            self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        elif self.provider == "gemini":
            from google import genai
            self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")


    @trace_llm_call
    def generate(self, request: LLMRequest) -> LLMResponse:
        """Routes the generic request to the active provider."""
        if self.provider == "openai":
            return self._call_openai(request)
        elif self.provider == "anthropic":
            return self._call_anthropic(request)
        elif self.provider == "gemini":
            return self._call_gemini(request)
        raise NotImplementedError()

    def _call_openai(self, request: LLMRequest) -> LLMResponse:
        messages = []
        for m in request.messages:
            msg = {"role": m.role, "content": m.content}
            if m.tool_calls:
                msg["tool_calls"] = [
                    {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
                    for tc in m.tool_calls
                ]
            if m.tool_call_id:
                msg["tool_call_id"] = m.tool_call_id
            messages.append(msg)
            
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": request.temperature,
        }
        
        if request.tools:
            kwargs["tools"] = [{"type": "function", "function": t} for t in request.tools]
        if request.response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}
            
        response = self.client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        
        tool_calls = None
        if choice.message.tool_calls:
            tool_calls = [
                ToolCall(id=tc.id, name=tc.function.name, arguments=json.loads(tc.function.arguments))
                for tc in choice.message.tool_calls
            ]
            
        return LLMResponse(
            content=choice.message.content or "",
            tool_calls=tool_calls,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            model_name=response.model
        )

    def _call_anthropic(self, request: LLMRequest) -> LLMResponse:
        # Extract system prompt (Anthropic handles it separately)
        system_msg = next((m.content for m in request.messages if m.role == "system"), "")
        messages = [{"role": m.role, "content": m.content} for m in request.messages if m.role != "system"]
        
        kwargs = {
            "model": self.model,
            "system": system_msg,
            "messages": messages,
            "max_tokens": 4096,
            "temperature": request.temperature,
        }
        
        if request.tools:
            # Map OpenAI tool schema to Anthropic format
            kwargs["tools"] = [
                {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "input_schema": t.get("parameters", {"type": "object", "properties": {}})
                } for t in request.tools
            ]
            
        response = self.client.messages.create(**kwargs)
        
        content = ""
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, arguments=block.input))
                
        return LLMResponse(
            content=content,
            tool_calls=tool_calls if tool_calls else None,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model_name=response.model
        )
        
    def _call_gemini(self, request: LLMRequest) -> LLMResponse:
        from google.genai import types
        
        # Extract system prompt
        system_instructions = next((m.content for m in request.messages if m.role == "system"), None)
        
        # Build contents from message history
        contents = []
        tool_name_map = {}
        for m in request.messages:
            if m.role == "system":
                continue
            elif m.role == "user":
                contents.append(types.Content(role="user", parts=[types.Part.from_text(text=m.content)]))
            elif m.role == "assistant":
                parts = []
                if m.content:
                    parts.append(types.Part.from_text(text=m.content))
                if m.tool_calls:
                    for tc in m.tool_calls:
                        tool_name_map[tc.id] = tc.name
                        if getattr(tc, "thought_signature", None):
                            parts.append(types.Part(
                                function_call=types.FunctionCall(name=tc.name, args=tc.arguments),
                                thought_signature=tc.thought_signature
                            ))
                        else:
                            parts.append(types.Part(
                                function_call=types.FunctionCall(name=tc.name, args=tc.arguments)
                            ))
                contents.append(types.Content(role="model", parts=parts))
            elif m.role == "tool":
                fn_name = tool_name_map.get(m.tool_call_id) or m.tool_call_id or "tool_result"
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part.from_function_response(
                        name=fn_name,
                        response={"result": m.content}
                    )]
                ))
        
        # Fallback to single text block if empty
        if not contents:
            contents = [types.Content(role="user", parts=[types.Part.from_text(text="")])]

        config = types.GenerateContentConfig(
            temperature=request.temperature,
            system_instruction=system_instructions,
        )
        if request.response_format == "json":
            config.response_mime_type = "application/json"
            
        if request.tools:
            fn_decls = [
                {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("parameters", {"type": "object", "properties": {}})
                }
                for t in request.tools
            ]
            config.tools = [{"function_declarations": fn_decls}]

        import time
        response = None
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config
                )
                break
            except Exception as e:
                err_str = str(e)
                if ("503" in err_str or "UNAVAILABLE" in err_str or "RemoteProtocolError" in err_str) and attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))
                    continue
                raise e
        
        content = ""
        tool_calls = []
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if getattr(part, "text", None):
                    content += part.text
                if getattr(part, "function_call", None):
                    fc = part.function_call
                    tool_calls.append(ToolCall(
                        id=getattr(fc, "id", None) or fc.name,
                        name=fc.name,
                        arguments=dict(fc.args or {}),
                        thought_signature=getattr(part, "thought_signature", None)
                    ))

        return LLMResponse(
            content=content,
            tool_calls=tool_calls if tool_calls else None,
            input_tokens=0,
            output_tokens=0,
            model_name=self.model
        )