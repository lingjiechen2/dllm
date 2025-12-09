from .configuration_llada import LLaDAConfig
from .modeling_llada import LLaDAModelLM
from .configuration_lladamoe import LLaDAMoEConfig
from .modeling_lladamoe import LLaDAMoEModelLM
from .configuration_llada2_moe import LLaDA2MoeConfig
from .modeling_llada2_moe import LLaDA2MoeModelLM

# Register with HuggingFace Auto classes for local usage
try:
    from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM

    AutoConfig.register("llada", LLaDAConfig)
    AutoModel.register(LLaDAConfig, LLaDAModelLM)
    AutoModelForMaskedLM.register(LLaDAConfig, LLaDAModelLM)

    AutoConfig.register("lladamoe", LLaDAMoEConfig)
    AutoModel.register(LLaDAMoEConfig, LLaDAMoEModelLM)
    AutoModelForMaskedLM.register(LLaDAMoEConfig, LLaDAMoEModelLM)

    AutoConfig.register("llada2_moe", LLaDA2MoeConfig)
    AutoModel.register(LLaDA2MoeConfig, LLaDA2MoeModelLM)
    AutoModelForMaskedLM.register(LLaDA2MoeConfig, LLaDA2MoeModelLM)
except ImportError:
    # transformers not available or Auto classes not imported
    pass
