# # launch the offline engine
# import asyncio

# import sglang as sgl
# import sglang.test.doc_patch
# from sglang.utils import async_stream_and_merge, stream_and_merge

# def main():
#     llm = sgl.Engine(model_path="qwen/qwen2.5-0.5b-instruct")

#     prompts = [
#         "Hello, my name is",
#         "The president of the United States is",
#         "The capital of France is",
#         "The future of AI is",
#     ]

#     sampling_params = {"temperature": 0.8, "top_p": 0.95}

#     outputs = llm.generate(prompts, sampling_params)
#     for prompt, output in zip(prompts, outputs):
#         print("===============================")
#         print(f"Prompt: {prompt}\nGenerated text: {output['text']}")


# if __name__ == "__main__":
#     main()


# import requests

# response = requests.post(
#     f"http://localhost:{port}/generate",
#     json={
#         "text": "The capital of France is",
#         "sampling_params": {
#             "temperature": 0,
#             "max_new_tokens": 32,
#             "step":32
#         },
#     },
# )

# print_highlight(response.json())



import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import dllm

# model_path = "/home/lingjie7/models/huggingface/inclusionAI/LLaDA2.0-mini"
model_path = "/home/lingjie7/models/huggingface/GSAI-ML/LLaDA-8B-Base"
device = "cuda:5"
model = AutoModelForCausalLM.from_pretrained(
    model_path, device_map=device
)
model = model.to(torch.bfloat16)
model.eval()
tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

prompt = "Why does Camus think that Sisyphus is happy?"
input_ids = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    add_generation_prompt=True,
    tokenize=True,
    return_tensors="pt",
)
generated_tokens = model.generate(
    inputs=input_ids,
    eos_early_stop=True,
    gen_length=512,
    block_length=32,
    steps=32,
    temperature=0.0,
)
generated_answer = tokenizer.decode(
    generated_tokens[0],
    skip_special_tokens=True,
)
breakpoint()
print(generated_answer)
