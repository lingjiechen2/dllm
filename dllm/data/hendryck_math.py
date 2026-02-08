from __future__ import annotations

from typing import Dict, List, Optional

from datasets import Dataset, DatasetDict, concatenate_datasets, get_dataset_config_names, load_dataset


def map_fn(example: Dict[str, str]) -> Dict[str, List[Dict[str, str]]]:
    return {
        "messages": [
            {"role": "user", "content": example["problem"]},
            {"role": "assistant", "content": example["solution"]},
        ]
    }


def _sample_first(ds: Dataset, n_examples: Optional[int]) -> Dataset:
    if n_examples is None:
        return ds
    if n_examples <= 0:
        return ds.select([])
    return ds.select(range(min(n_examples, len(ds))))


def load_dataset_math(dataset_path: str, n_examples: Optional[int] = None) -> DatasetDict:
    configs = get_dataset_config_names(dataset_path)

    train_parts: List[Dataset] = []
    test_parts: List[Dataset] = []

    for cfg in configs:
        dd = load_dataset(dataset_path, cfg)
        for split, parts in (("train", train_parts), ("test", test_parts)):
            ds = _sample_first(dd[split], n_examples)
            ds = ds.map(map_fn, remove_columns=ds.column_names)
            parts.append(ds)

    train = concatenate_datasets(train_parts) if len(train_parts) > 1 else train_parts[0]
    test = concatenate_datasets(test_parts) if len(test_parts) > 1 else test_parts[0]
    return DatasetDict({"train": train, "test": test})


if __name__ == "__main__":
    dataset_path = "/mnt/lustrenew/mllm_aligned/shared/datasets/huggingface/EleutherAI/hendrycks_math"
    dd = load_dataset_math(dataset_path, n_examples=3)
    breakpoint()
    print(dd)
    print(dd["train"].column_names)
    print(dd["train"][0]["messages"])
