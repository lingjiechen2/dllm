"""
Quick sanity test: load A2DModel with trust_remote_code=True and run a short diffusion-style generate.

Usage:
    python3 test_a2d_generate.py --model-path A2DModel --steps 6 --masks 4
"""
import argparse
import copy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="A2DModel", help="Path or repo id for the A2D model folder.")
    parser.add_argument("--steps", type=int, default=6, help="Diffusion steps (kept small for a quick check).")
    parser.add_argument("--masks", type=int, default=4, help="Number of <|mask|> tokens to fill.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # Build a tiny masked prompt. Diffusion generation expects <|mask|> tokens to fill.
    prompt = "Hello, this is a quick A2D generation test " + ("<|mask|>" * args.masks)
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map="auto" if device.type == "cuda" else None,
    )
    model.eval()

    # Clone config so we can tweak without mutating the saved file.
    gen_cfg = copy.deepcopy(model.generation_config)
    if gen_cfg.mask_token_id is None and getattr(tokenizer, "mask_token_id", None) is not None:
        gen_cfg.mask_token_id = tokenizer.mask_token_id
    gen_cfg.steps = args.steps
    gen_cfg.max_length = input_ids.shape[1]  # keep sequence length short for the smoke test
    gen_cfg.return_dict_in_generate = True
    gen_cfg.output_history = False

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=gen_cfg,
        )

    sequences = output.sequences if hasattr(output, "sequences") else output
    text = tokenizer.batch_decode(sequences, skip_special_tokens=False)
    print("\n=== Generated ===")
    for idx, t in enumerate(text):
        print(f"[{idx}] {t}")


if __name__ == "__main__":
    main()
