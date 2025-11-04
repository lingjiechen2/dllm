"""
python -u examples/bert/eval_lambada.py --model_name_or_path "YOUR_MODEL_PATH"
"""

import json
import string
import torch
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
from sacremoses import MosesDetokenizer
from transformers import BasicTokenizer, HfArgumentParser
import unicodedata

import dllm
from dllm.pipelines import llada
from dllm.tools.chat import decode_trim

# ============================================================
# 1. Arguments and Configuration
# ============================================================

@dataclass
class ScriptArguments:
    model_name_or_path: str = "ModernBERT-large/tulu-3-smoltalk/epochs-10-bs-384-len-1024/checkpoint-final"
    seed: int = 42
    limit: int = 200  # for debugging
    def __post_init__(self):
        self.model_name_or_path = dllm.utils.resolve_with_base_env(
            self.model_name_or_path, "BASE_MODELS_DIR"
        )

@dataclass
class GeneratorConfig(llada.LLaDAGeneratorConfig):
    steps: int = 128
    max_new_tokens: int = 128
    block_length: int = 64
    temperature: float = 0.0
    remasking: str = "random"

parser = HfArgumentParser((ScriptArguments, GeneratorConfig))
script_args, gen_config = parser.parse_args_into_dataclasses()
torch.manual_seed(script_args.seed)

# ============================================================
# 2. Load Model and Tokenizer
# ============================================================

model = dllm.utils.get_model(model_args=script_args).eval()
tokenizer = dllm.utils.get_tokenizer(model_args=script_args)
generator = llada.LLaDAGenerator(model=model, tokenizer=tokenizer)
basic_tokenizer = BasicTokenizer(do_lower_case=False)
detokenizer = MosesDetokenizer(lang='en')

# ============================================================
# 3. Utility Functions
# ============================================================

def detokenize(line):
    toks = line.split()
    return detokenizer.detokenize(toks)

def preprocess(text):
    text = text.replace("“", '"').replace("”", '"').replace("''", '"').replace("``", '"')
    return '\n' + text.strip()

def remove_last_word(line):
    line = line.strip()
    toks = basic_tokenizer.tokenize(line)
    length_of_word = len(toks[-1])
    assert length_of_word > 0
    return line[:-length_of_word].strip(), toks[-1]


def is_punctuation_token(token: str) -> bool:
    """
    Return True if the token consists entirely of punctuation characters.
    Includes both ASCII (string.punctuation) and Unicode punctuation such as “ ” — … etc.
    """
    token = token.strip()
    if not token:
        return False

    # Quick path: all ASCII punctuation
    if all(ch in string.punctuation for ch in token):
        return True

    # Unicode-aware path: check punctuation category
    if all(unicodedata.category(ch).startswith("P") for ch in token):
        return True

    # Fallback: use BasicTokenizer for combined tokens like '.”'
    toks = basic_tokenizer.tokenize(token)
    return all(
        all(ch in string.punctuation or unicodedata.category(ch).startswith("P") for ch in t)
        for t in toks
    )

# ============================================================
# 4. BERT Prediction Logic
# ============================================================

basic_tokenizer = BasicTokenizer(do_lower_case=False)

stopwords = {
    'ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about',
    'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be',
    'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself',
    'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each',
    'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his',
    'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this',
    'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours',
    'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and',
    'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that',
    'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under',
    'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself',
    'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs',
    'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here',
    'than'
}

stopwords.update(set(string.punctuation))

@torch.no_grad()
def bert_predict(context, beam_width=128, max_predictions=6):
    """
    Iteratively predict up to `max_predictions` tokens using BERT's masked LM head.
    Each step fills the final [MASK], appends the best token, and adds a new [MASK].
    Performs beam search (width=128) and stopword/punctuation filtering.
    """
    generated_tokens = []

    for _ in range(max_predictions):
        # Add [MASK] to current context
        masked_input = context.strip() + " [MASK]"
        inputs = tokenizer(masked_input, return_tensors="pt").to(model.device)

        # Locate [MASK] position
        mask_index = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1].item()

        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits[0, mask_index]

        # Top-k beam candidates
        topk_logits, topk_indices = torch.topk(logits, k=beam_width, dim=-1)
        topk_indices = topk_indices.tolist()

        # Select first valid token (not stopword/punctuation)
        predicted_token_id = None
        for idx in topk_indices:
            decoded_token = tokenizer.decode([idx]).strip()
            lower_tok = decoded_token.lower()
            if lower_tok and lower_tok not in stopwords and not is_punctuation_token(lower_tok):
                predicted_token_id = idx
                break

        # Fallback if everything filtered out
        if predicted_token_id is None:
            predicted_token_id = topk_indices[0]

        # Decode, append, and update context
        predicted_token = tokenizer.decode([predicted_token_id]).strip()
        generated_tokens.append(predicted_token)

        # If the predicted token seems terminal (punctuation), stop early
        if is_punctuation_token(predicted_token):
            break

        # Extend the context for next iteration
        context = context.strip() + " " + predicted_token

    # Return the entire decoded continuation
    return " ".join(generated_tokens).strip()


def predict_wrapper(line):
    context, last_word = remove_last_word(line)
    context = "\n" + context

    try:
        # Multi-token generation
        prediction = bert_predict(context, beam_width=128, max_predictions=6)
        predicted_part = prediction.strip()
        print(f"predicted part:{predicted_part}, ground truth:{last_word}")
        # Extract the first valid predicted word using BasicTokenizer
        tokens = basic_tokenizer.tokenize(predicted_part)
        if tokens:
            predicted_word = tokens[0]
        else:
            print(f"[WARN] Empty predicted_part: {repr(predicted_part)}")
            predicted_word = ""

    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        predicted_word = ""

    return predicted_word, last_word


# ============================================================
# 5. Evaluation Loop
# ============================================================

with open("lambada_test.jsonl") as f:
    lines = [json.loads(line)["text"] for line in f]
    lines = [preprocess(line) for line in lines]

errors, total = 0, 0
limit = script_args.limit

for i, line in enumerate(tqdm(lines, desc="Evaluating BERT on LAMBADA", unit="line")):
    line = detokenize(line)
    line = line.replace(" n't", "n't")
    line = '\n' + line

    predicted_word, last_word = predict_wrapper(line)
    is_error = predicted_word.lower() != last_word.lower()

    if is_error:
        errors += 1
    total += 1

    if i >= limit:
        break

print(f"{i+1:5d}/{len(lines):5d}, acc: {100*(1 - errors/total):.2f}")
