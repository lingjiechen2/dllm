"""
LLaDA Fast-dLLM configuration wrapper.
Reuses LLaDAConfig but registers under a different model_type.
"""
from .configuration_llada import LLaDAConfig


class LLaDAFastDLLMConfig(LLaDAConfig):
    model_type = "llada_fastdllm"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
