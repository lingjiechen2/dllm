from .configuration_dream import DreamConfig
from .configuration_fastdllmdream import FastDLLMDreamConfig
from .modeling_dream import DreamModel
from .modeling_fastdllmdream import FastDLLMDreamModel

# Register with HuggingFace Auto classes for local usage
try:
    from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM

    AutoConfig.register("Dream", DreamConfig)
    AutoModel.register(DreamConfig, DreamModel)
    AutoModelForMaskedLM.register(DreamConfig, DreamModel)

    AutoConfig.register("fastdllm_Dream", FastDLLMDreamConfig)
    AutoModel.register(FastDLLMDreamConfig, FastDLLMDreamModel)
    AutoModelForMaskedLM.register(FastDLLMDreamConfig, FastDLLMDreamModel)
except ImportError:
    # transformers not available or Auto classes not imported
    pass
