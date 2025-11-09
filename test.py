# """
# python -u examples/dream/generate.py --model_name_or_path "YOUR_MODEL_PATH"
# """

# from dataclasses import dataclass

# import transformers

# import dllm
# from dllm.tools.chat import decode_trim
# from dllm.pipelines import dream

# # import torch
# # torch.manual_seed(0)
# # torch.cuda.manual_seed_all(0)
# # torch.use_deterministic_algorithms(True)

# @dataclass
# class ScriptArguments:
#     model_name_or_path: str = "Dream-org/Dream-v0-Base-7B"
#     seed: int = 0
#     visualize: bool = True
#     def __post_init__(self):
#         self.model_name_or_path = dllm.utils.resolve_with_base_env(
#             self.model_name_or_path, "BASE_MODELS_DIR"
#         )

# @dataclass
# class GeneratorConfig(dream.DreamGeneratorConfig):
#     steps: int = 128
#     max_new_tokens: int = 128
#     temperature: float = 0.0
#     top_p: float = 0.95
#     alg: str = "entropy"
#     alg_temp: float = 0.0


# parser = transformers.HfArgumentParser(
#     (ScriptArguments, GeneratorConfig)
# )
# script_args, gen_config = parser.parse_args_into_dataclasses()
# transformers.set_seed(script_args.seed)

# # Load model & tokenizer
# model = dllm.utils.get_model(model_args=script_args).eval()
# tokenizer = dllm.utils.get_tokenizer(model_args=script_args)
# generator = dream.DreamGenerator(model=model, tokenizer=tokenizer)


# text = "You are an expert Python programmer, and here is your task: Write a function to find the similar elements from the given two tuple lists. Your code should pass these tests:\n\nassert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)\nassert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)\nassert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)\n[BEGIN]\ndef similar_elements(test_tup1, test_tup2):\r\n  res = tuple(set(test_tup1) & set(test_tup2))\r\n  return (res) \n[DONE]\n\nYou are an expert Python programmer, and here is your task: Write a python function to identify non-prime numbers. Your code should pass these tests:\n\nassert is_not_prime(2) == False\nassert is_not_prime(10) == True\nassert is_not_prime(35) == True\n[BEGIN]\nimport math\r\ndef is_not_prime(n):\r\n    result = False\r\n    for i in range(2,int(math.sqrt(n)) + 1):\r\n        if n % i == 0:\r\n            result = True\r\n    return result\n[DONE]\n\nYou are an expert Python programmer, and here is your task: Write a function to find the largest integers from a given list of numbers using heap queue algorithm. Your code should pass these tests:\n\nassert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65] \nassert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75] \nassert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35]\n[BEGIN]\nimport heapq as hq\r\ndef heap_queue_largest(nums,n):\r\n  largest_nums = hq.nlargest(n, nums)\r\n  return largest_nums\n[DONE]\n\nYou are an expert Python programmer, and here is your task: Write a python function to remove first and last occurrence of a given character from the string. Your code should pass these tests:\n\nassert remove_Occ(\"hello\",\"l\") == \"heo\"\nassert remove_Occ(\"abcda\",\"a\") == \"bcd\"\nassert remove_Occ(\"PHP\",\"P\") == \"H\"\n[BEGIN]\n"
# inputs = tokenizer(text, return_tensors="pt")
# inputs['input_ids'] = inputs['input_ids'].to(model.device)
# inputs['attention_mask'] = inputs['attention_mask'].to(model.device)
# inputs = inputs['input_ids']




# outputs = generator.generate(inputs, gen_config, return_dict_in_generate=True)
# sequences = decode_trim(tokenizer, outputs.sequences.tolist(), inputs)

# for iter, s in enumerate(sequences):
#     print("\n" + "-" * 80)
#     print(f"[Case {iter}]")
#     print("-" * 80)
#     print(s.strip() if s.strip() else "<empty>")
# print("\n" + "=" * 80 + "\n")

# breakpoint()



# # output = model.diffusion_generate(
# #     inputs,
# #     max_new_tokens=128,
# #     output_history=True,
# #     return_dict_in_generate=True,
# #     steps=128,
# #     temperature=0.0,
# #     top_p=0.95,
# #     alg="entropy",
# #     alg_temp=0.,
# # )
# # # breakpoint()
# # print(tokenizer.decode(output.sequences[0]))



# # import torch
# # torch.save(probs, "generation_utils.pt")
# # torch.save(probs, "generator.pt")

# # generator = torch.load("generator.pt", map_location=torch.device("cpu"))
# # generation_utils = torch.load("generation_utils.pt", map_location=torch.device("cpu"))
# # torch.isclose(generator, generation_utils).all()


from transformers import AutoTokenizer, AutoModelForMaskedLM

model_id = "/mnt/lustrenew/mllm_aligned/shared/models/huggingface/answerdotai/ModernBERT-base/"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForMaskedLM.from_pretrained(model_id).to("cuda")

text = "from typing import List\n\n\ndef mean_absolute_deviation(numbers: List[float]) -> float:\n    \"\"\" For a given list of input numbers, calculate Mean Absolute Deviation\n    around the mean of this dataset.\n    Mean Absolute Deviation is the average absolute difference between each\n    element and a centerpoint (mean in this case):\n    MAD = average | x - x_mean |\n    >>> mean_absolute_deviation([1.0, 2.0, 3.0, 4.0])\n    1.0\n    \"\"\"\n" + "[MASK]" * 128
inputs = tokenizer(text, return_tensors="pt").to("cuda")
outputs = model(**inputs)

# To get predictions for the mask:
masked_index = inputs["input_ids"][0].tolist().index(tokenizer.mask_token_id)
predicted_token_id = outputs.logits[0, masked_index].argmax(axis=-1)
predicted_token = tokenizer.decode(predicted_token_id)
print("Predicted token:", predicted_token)