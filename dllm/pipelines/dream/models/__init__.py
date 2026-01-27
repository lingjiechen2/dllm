from .configuration_dream import DreamConfig
from .modeling_dream import DreamModel
from .configuration_dream_fastdllm import DreamFastDLLMConfig
from .modeling_dream_fastdllm import DreamFastDLLMModel

# Register with HuggingFace Auto classes for local usage
try:
    from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM

    AutoConfig.register("Dream", DreamConfig)
    AutoModel.register(DreamConfig, DreamModel)
    AutoModelForMaskedLM.register(DreamConfig, DreamModel)

    AutoConfig.register("Dream_fastdllm", DreamFastDLLMConfig)
    AutoModel.register(DreamFastDLLMConfig, DreamFastDLLMModel)
    AutoModelForMaskedLM.register(DreamFastDLLMConfig, DreamFastDLLMModel)
except ImportError:
    # transformers not available or Auto classes not imported
    pass
