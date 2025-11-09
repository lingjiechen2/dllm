import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, BasicTokenizer
from datasets import load_dataset
from tqdm import tqdm
import string
import json

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "/mnt/lustrenew/mllm_aligned/shared/models/huggingface/openai-community/gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name).to(device).eval()
basic_tokenizer = BasicTokenizer(do_lower_case=False)

from sacremoses import MosesTokenizer, MosesDetokenizer
detokenizer = MosesDetokenizer(lang='en')

def detokenize(line):
    toks = line.split()
    return detokenizer.detokenize(toks)

# Remove all smart quotes
def preprocess(text):
    text = text.replace("“", '"')
    text = text.replace("”", '"')
    text = text.replace("''", '"')
    text = text.replace("``", '"')
    return '\n'+text.strip()

# def remove_last_word(line):
#     """
#     Removes the last word (alphanumeric token) from a sentence and returns:
#     (context_without_last_word, last_word)
#     Robust to punctuatison and spacing.
#     """
#     line = line.strip()
#     words = line.split()
#     if not words:
#         return "", ""

#     last_word = words[-1].strip(string.punctuation)
#     context = " ".join(words[:-1]).strip()
#     return context, last_word

def remove_last_word(line):
  line = line.strip()
  toks = basic_tokenizer.tokenize(line)
  length_of_word = len(toks[-1])
  assert length_of_word>0
  return line[:-length_of_word].strip(), toks[-1]


stopwords = {'ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than'}

def predict(line, max_predictions=6, beam_width=128):
    """
    Autoregressively generate continuation for a given line using GPT-2.
    Returns the full line + predicted continuation.
    """
    # Tokenize input and move to device
    inputs = tokenizer(line, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    input_list = input_ids[0].tolist()
    past_key_values = None

    # Autoregressive loop (manual generation)
    for _ in range(max_predictions):
        with torch.no_grad():
            outputs = model(input_ids=input_ids, past_key_values=past_key_values)
            logits = outputs.logits
            past_key_values = outputs.past_key_values

        # Select top-k candidates for current step
        topk_logits, topk_indices = torch.topk(logits[:, -1, :], k=beam_width, dim=-1)

        # Decode candidates and filter stopwords
        topk_indices = topk_indices[0].tolist()
        predicted_token_id = None
        for idx in topk_indices:
            decoded_token = tokenizer.decode([idx]).strip()
            if decoded_token not in stopwords and decoded_token != "":
                predicted_token_id = idx
                break

        # Fallback if all were stopwords
        if predicted_token_id is None:
            predicted_token_id = topk_indices[0]

        # Append predicted token and continue
        input_list.append(predicted_token_id)
        input_ids = torch.tensor([[predicted_token_id]], device=device)
    # Decode the final sequence
    return tokenizer.decode(input_list)

def predict_wrapper(line):
    context, last_word = remove_last_word(line)
    context = "\n"+context
    prediction = predict(context, 6)
    predicted_part = prediction[len(context):].strip()
    try:
        tokens = basic_tokenizer.tokenize(predicted_part)
        if tokens:
            predicted_word = tokens[0]
        else:
            print(f"[WARN] Empty predicted_part: {repr(predicted_part)}")
            predicted_word = ""
    except Exception as e:
        print(f"[ERROR] Tokenization failed: {e}")
        print(f"predicted_part raw repr: {repr(predicted_part)}")
        predicted_word = ""
    return predicted_word, last_word
    

with open("lambada_test.jsonl") as f:
    lines = [json.loads(line)["text"] for line in f]
    lines = [preprocess(line) for line in lines]

errors = 0
total = 0
limit = 6000

for i, line in enumerate(tqdm(lines, desc="Evaluating LAMBADA", unit="line")):
    line = line.strip()
    line = detokenizer.detokenize(line.split())
    line = line.replace(" n't", "n't")
    line = '\n'+line
    predicted_word, last_word = predict_wrapper(line)
    is_error = predicted_word.lower() != last_word.lower()
    
    if is_error:
        errors += 1
    total+=1

    if i > limit:
        break

print(f"{i:5d}/{len(lines):5d}, acc: {100*(1-errors/total):.2f}")