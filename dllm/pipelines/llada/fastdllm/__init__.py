from .configuration_llada_fastdllm import LLaDAFastdLLMConfig
from .modeling_llada_fastdllm import LLaDAFastdLLMModelLM
from .sampler_fastdllm import LLaDAFastdLLMSampler, LLaDAFastdLLMSamplerConfig

# Optional: register with transformers Auto classes when available
try:
    from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM

    AutoConfig.register("llada_fastdllm", LLaDAFastdLLMConfig)
    AutoModel.register(LLaDAFastdLLMConfig, LLaDAFastdLLMModelLM)
    AutoModelForMaskedLM.register(LLaDAFastdLLMConfig, LLaDAFastdLLMModelLM)
except ImportError:
    pass

__all__ = [
    "LLaDAFastdLLMConfig",
    "LLaDAFastdLLMModelLM",
    "LLaDAFastdLLMSampler",
    "LLaDAFastdLLMSamplerConfig",
]
