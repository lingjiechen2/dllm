#!/usr/bin/env python3
"""
Build a dataset of prompts, each bundling a random subset of edges with all nodes involved.

Examples:
- PT style (single text field): python3 scripts/build_prompt.py --edges edges.jsonl --nodes nodes.jsonl --edge-count 5 --style pt --output tmp/prompts.pt.jsonl
- Instruction style (chat format): python3 scripts/build_prompt.py --edges edges.jsonl --nodes nodes.jsonl --edge-count 5 --style instruction --output tmp/prompts.instruction.jsonl
- Save directly to disk: python3 scripts/build_prompt.py --edge-count 5 --style pt --save-to-disk /home/lingjie7/datasets/tmp/food_trig/food_category
"""

import argparse
import json
import random
from math import ceil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from datasets import Dataset


def load_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} in {path}: {exc}") from exc


def index_nodes(nodes_path: Path) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    by_label: Dict[str, dict] = {}
    by_id: Dict[str, dict] = {}
    for node in load_jsonl(nodes_path):
        node_id = node.get("id")
        label = node.get("label", node_id)
        if node_id is None:
            raise ValueError(f"Node missing 'id': {node}")
        by_id[str(node_id)] = node
        if label:
            by_label[str(label)] = node
    return by_label, by_id


def load_edges(edges_path: Path) -> List[dict]:
    return [edge for edge in load_jsonl(edges_path)]


def collect_nodes_for_edges(
    chosen_edges: List[dict], nodes_by_label: Dict[str, dict], nodes_by_id: Dict[str, dict]
) -> List[dict]:
    seen = {}
    for edge in chosen_edges:
        for key in ("source", "target"):
            value = edge.get(key)
            if value is None:
                continue
            value_str = str(value)
            node = nodes_by_label.get(value_str) or nodes_by_id.get(value_str)
            if node is None:
                # Fallback node if not present in provided nodes.jsonl
                node = {"id": value_str}
            if node["id"] not in seen:
                seen[node["id"]] = {"id": node["id"]}
    return list(seen.values())


def build_prompt(nodes: List[dict], edges: List[dict]) -> dict:
    return {"nodes": nodes, "edges": edges}


def format_prompt(prompt: dict, style: str) -> dict:
    if style == "pt":
        # Store the full graph JSON as a single text field
        return {"text": json.dumps(prompt, ensure_ascii=False)}

    if style == "instruction":
        nodes_only = json.dumps({"nodes": prompt["nodes"]}, ensure_ascii=False)
        full_graph = json.dumps(prompt, ensure_ascii=False)
        user_msg = (
            "Complete the graph from the partial JSON below. Only the node list is provided; "
            'infer all missing edges. Return ONLY the JSON with keys "nodes" and "edges".\n\n'
            f"Partial graph (nodes-only):\n{nodes_only}"
        )
        assistant_msg = full_graph
        return {
            "messages": [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ]
        }

    raise ValueError(f"Unknown style: {style}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create prompts that contain nodes plus randomly chosen, non-overlapping edges."
        )
    )
    parser.add_argument("--nodes", default="nodes.jsonl", help="Path to nodes.jsonl input.")
    parser.add_argument("--edges", default="edges.jsonl", help="Path to edges.jsonl input.")
    parser.add_argument(
        "--edge-count",
        type=int,
        default=5,
        help="How many edges to include in the prompt (uses all if fewer available).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed for reproducible sampling.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help=(
            "Number of prompts to generate. Default chunks all edges with size edge-count "
            "(last chunk may be smaller)."
        ),
    )
    parser.add_argument(
        "--output",
        default="-",
        help="Where to write prompts as JSONL. Use '-' for stdout.",
    )
    parser.add_argument(
        "--style",
        choices=["pt", "instruction"],
        default="pt",
        help="Output format. 'pt' -> single 'text' field; 'instruction' -> single 'message' field.",
    )
    parser.add_argument(
        "--save-to-disk",
        dest="save_to_disk",
        default=None,
        help="Optional path to save the dataset with Hugging Face save_to_disk.",
    )
    args = parser.parse_args()

    if args.edge_count <= 0:
        raise ValueError("edge-count must be positive")
    if args.samples is not None and args.samples <= 0:
        raise ValueError("samples must be positive when provided")

    nodes_by_label, nodes_by_id = index_nodes(Path(args.nodes))
    edges = load_edges(Path(args.edges))

    rng = random.Random(args.seed)
    shuffled_edges = edges.copy()
    rng.shuffle(shuffled_edges)

    max_samples = ceil(len(shuffled_edges) / args.edge_count)
    sample_count = args.samples if args.samples is not None else max_samples
    if sample_count > max_samples:
        raise ValueError(
            f"Not enough edges for {sample_count} samples of size {args.edge_count}. "
            f"Maximum available samples: {max_samples}"
        )

    prompts: List[dict] = []
    for idx in range(sample_count):
        start = idx * args.edge_count
        end = start + args.edge_count
        edge_slice = shuffled_edges[start:end]
        if not edge_slice:
            break
        prompt_nodes = collect_nodes_for_edges(edge_slice, nodes_by_label, nodes_by_id)
        prompts.append(build_prompt(prompt_nodes, edge_slice))

    formatted_prompts = [format_prompt(prompt, args.style) for prompt in prompts]

    out_path = Path(args.output)
    lines = "\n".join(json.dumps(item, ensure_ascii=False) for item in formatted_prompts)
    if args.output == "-":
        print(lines)
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(lines, encoding="utf-8")
        print(
            f"Wrote {len(prompts)} prompts (edge-count={args.edge_count}, style={args.style}) to {out_path}"
        )

    if args.save_to_disk:
        ds = Dataset.from_list(formatted_prompts)
        save_path = Path(args.save_to_disk)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(save_path))
        print(
            f"Saved dataset with {len(formatted_prompts)} prompts to {save_path} (style={args.style})"
        )


if __name__ == "__main__":
    main()
