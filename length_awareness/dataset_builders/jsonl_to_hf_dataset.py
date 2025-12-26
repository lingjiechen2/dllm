#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Value


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Convert a multi-granularity JSONL (question + budgets[]) into a flat HuggingFace "
            "Dataset/DatasetDict (one row per answer) and save_to_disk."
        )
    )
    p.add_argument(
        "--input",
        required=True,
        help="Path to input .jsonl file.",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Output directory to write with save_to_disk().",
    )
    p.add_argument(
        "--split",
        default="train",
        help="Split name to store in a DatasetDict (default: train).",
    )
    p.add_argument(
        "--datasetdict",
        action="store_true",
        help="Save as DatasetDict with the provided --split key (recommended).",
    )
    p.add_argument(
        "--no-datasetdict",
        dest="datasetdict",
        action="store_false",
        help="Save as a plain Dataset instead of DatasetDict.",
    )
    p.set_defaults(datasetdict=True)
    return p.parse_args()


def row_generator(input_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            question = obj.get("question")
            budgets = obj.get("budgets") or []
            if question is None:
                raise ValueError(f"Missing 'question' at line {line_num} in {input_path}")
            if not isinstance(budgets, list):
                raise ValueError(
                    f"Expected 'budgets' to be a list at line {line_num} in {input_path}"
                )

            for b in budgets:
                if not isinstance(b, dict):
                    continue
                answer = b.get("text")
                target_length = b.get("target_length", b.get("L"))
                if answer is None or target_length is None:
                    continue
                yield {
                    "question": question,
                    "answer": answer,
                    "target_length": int(target_length),
                }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(str(input_path))

    features = Features(
        {
            "question": Value("string"),
            "answer": Value("string"),
            "target_length": Value("int32"),
        }
    )
    ds = Dataset.from_generator(
        row_generator,
        gen_kwargs={"input_path": str(input_path)},
        features=features,
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if args.datasetdict:
        DatasetDict({args.split: ds}).save_to_disk(str(output_dir))
    else:
        ds.save_to_disk(str(output_dir))

    print(f"Loaded {ds.num_rows} rows from {input_path}")
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
