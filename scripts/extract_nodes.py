#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple


def load_edges(edges_path: Path) -> Iterable[Tuple[int, Dict[str, str]]]:
    with edges_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield line_no, json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc


def collect_nodes(edges_path: Path) -> Dict[str, Dict[str, object]]:
    nodes: Dict[str, Dict[str, object]] = {}
    for line_no, edge in load_edges(edges_path):
        for key in ("source", "target"):
            value = edge.get(key)
            if value is None:
                continue
            if not isinstance(value, str):
                raise ValueError(
                    f"Expected a string for '{key}' on line {line_no}, got {type(value)}"
                )
            if value not in nodes:
                nodes[value] = {
                    "id": value,
                    "label": value,
                    "types": ["Class"],
                    "source": "graph",
                }
    return nodes


def write_nodes(nodes: Dict[str, Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for value in sorted(nodes):
            json.dump(nodes[value], handle, ensure_ascii=False)
            handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract unique nodes from an edges JSONL file and write them to nodes.jsonl."
        )
    )
    parser.add_argument(
        "--edges",
        default="edges.jsonl",
        help="Path to the input edges.jsonl file.",
    )
    parser.add_argument(
        "--output",
        default="nodes.jsonl",
        help="Where to write the extracted nodes JSONL.",
    )
    args = parser.parse_args()

    edges_path = Path(args.edges)
    output_path = Path(args.output)

    nodes = collect_nodes(edges_path)
    write_nodes(nodes, output_path)
    print(f"Wrote {len(nodes)} nodes to {output_path}")


if __name__ == "__main__":
    main()
