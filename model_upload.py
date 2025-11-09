from huggingface_hub import HfApi

api = HfApi(token="hf-token")
api.upload_folder(
    folder_path="/path/to/local/model",
    repo_id="dllm-collection/ModernBERT-large-chat-v1",
    repo_type="model",
)
