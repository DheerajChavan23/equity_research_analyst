import os
from functools import wraps
from dotenv import load_dotenv

load_dotenv()

from src.llm.schema import LLMRequest, LLMResponse

# Support both Langfuse v4 (langfuse.observe) and v2/v3 (langfuse.decorators)
try:
    from langfuse import Langfuse, observe
    langfuse_client = Langfuse()
    USE_V4 = True
except ImportError:
    try:
        from langfuse import Langfuse
        from langfuse.decorators import observe, langfuse_context
        langfuse_client = Langfuse()
        USE_V4 = False
    except ImportError:
        observe = None
        langfuse_client = None
        USE_V4 = False

def trace_llm_call(func):
    """
    Decorator for the LLMGateway.generate() method.
    Automatically logs prompts, responses, tools, and token costs.
    """
    obs_dec = observe(as_type="generation") if callable(observe) else (lambda f: f)

    @wraps(func)
    @obs_dec
    def wrapper(self, request: LLMRequest, *args, **kwargs):
        # 1. Log the incoming prompt and tools
        if langfuse_client is not None:
            try:
                metadata = {
                    "provider": self.provider,
                    "temperature": request.temperature,
                    "tools_provided": [t["name"] for t in request.tools] if request.tools else []
                }
                if USE_V4:
                    langfuse_client.update_current_generation(
                        input=[m.model_dump() for m in request.messages],
                        model=self.model,
                        metadata=metadata
                    )
                else:
                    langfuse_context.update_current_observation(
                        input=[m.model_dump() for m in request.messages],
                        model=self.model,
                        metadata=metadata
                    )
            except Exception:
                pass

        # 2. Execute the actual LLM call
        response: LLMResponse = func(self, request, *args, **kwargs)

        # 3. Log the output and token usage
        if langfuse_client is not None:
            try:
                output_val = response.content if response.content else "Tool Call Invoked"
                if USE_V4:
                    langfuse_client.update_current_generation(
                        output=output_val,
                        usage_details={
                            "input": response.input_tokens,
                            "output": response.output_tokens
                        }
                    )
                else:
                    langfuse_context.update_current_observation(
                        output=output_val,
                        usage={
                            "input": response.input_tokens,
                            "output": response.output_tokens
                        }
                    )
            except Exception:
                pass

        return response
    return wrapper