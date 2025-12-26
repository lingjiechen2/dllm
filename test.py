#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert JSON-LD into:
1) edges.jsonl: {"source": "...", "relation": "...", "target": "..."}
2) nodes.jsonl: {"id": "...", "label": "...", "types": [...], "source": "graph|edge_only"}

Optimization:
- Try multiple label properties (rdfs:label, schema:name, skos:prefLabel, dcterms:title, foaf:name)
- Drop nodes whose final label looks like an index (pure digits), from nodes.jsonl
"""

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# --- Common IRIs ---
RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"

# Extra "name" properties often used in KGs
SCHEMA_NAME = "http://schema.org/name"
SKOS_PREFLABEL = "http://www.w3.org/2004/02/skos/core#prefLabel"
DCT_TITLE = "http://purl.org/dc/terms/title"
FOAF_NAME = "http://xmlns.com/foaf/0.1/name"

LABEL_CANDIDATES = [
    RDFS_LABEL,
    SKOS_PREFLABEL,
    SCHEMA_NAME,
    DCT_TITLE,
    FOAF_NAME,
]

DEFAULT_EXCLUDE_PREDICATES = {
    RDF_TYPE,
    RDFS_LABEL,
    SKOS_PREFLABEL,
    SCHEMA_NAME,
    DCT_TITLE,
    FOAF_NAME,
    "http://nlp_microbe.ccnu.edu.cn/pop/statement#hasIdentifier",
}

DIGITS_ONLY = re.compile(r"^\d+$")


def local_name(uri: str) -> str:
    if not uri:
        return uri
    uri = uri.rstrip("/")
    if "#" in uri:
        frag = uri.split("#")[-1]
        return frag if frag else uri
    return uri.rsplit("/", 1)[-1] if "/" in uri else uri


def load_jsonld_nodes(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    nodes: List[Dict[str, Any]] = []

    def extract(obj: Any) -> None:
        nonlocal nodes
        if isinstance(obj, dict):
            if "@graph" in obj and isinstance(obj["@graph"], list):
                for n in obj["@graph"]:
                    if isinstance(n, dict):
                        nodes.append(n)
            else:
                if "@id" in obj:
                    nodes.append(obj)
        elif isinstance(obj, list):
            for x in obj:
                extract(x)

    extract(data)
    return nodes


def first_literal_str(x: Any) -> Optional[str]:
    """Extract first literal '@value' string from JSON-LD value structure."""
    if isinstance(x, list):
        for it in x:
            v = first_literal_str(it)
            if v:
                return v
        return None
    if isinstance(x, dict):
        v = x.get("@value")
        if isinstance(v, str) and v.strip():
            return v.strip()
        return None
    if isinstance(x, str) and x.strip():
        return x.strip()
    return None


def extract_types(node: Dict[str, Any]) -> List[str]:
    t = node.get("@type")
    types: List[str] = []
    if isinstance(t, list):
        for it in t:
            if isinstance(it, str):
                types.append(local_name(it))
    elif isinstance(t, str):
        types.append(local_name(t))
    seen = set()
    out = []
    for x in types:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def best_label_for_node(node: Dict[str, Any]) -> Optional[str]:
    """Try multiple label predicates; return the first good one."""
    for pred in LABEL_CANDIDATES:
        v = first_literal_str(node.get(pred))
        if v:
            return v
    return None


def build_id2label(nodes: List[Dict[str, Any]]) -> Dict[str, str]:
    m: Dict[str, str] = {}
    for node in nodes:
        nid = node.get("@id")
        if not isinstance(nid, str):
            continue
        lbl = best_label_for_node(node)
        if lbl:
            m[nid] = lbl
    return m


def is_entity_ref(x: Any) -> Optional[str]:
    if isinstance(x, dict) and isinstance(x.get("@id"), str):
        return x["@id"]
    if isinstance(x, str) and x.startswith("http"):
        return x
    return None


def readable_pred(pid: str, id2label: Dict[str, str]) -> str:
    return id2label.get(pid, local_name(pid))


def readable_node_label(nid: str, id2label: Dict[str, str]) -> str:
    """Prefer known label; fallback to local_name(uri)."""
    return id2label.get(nid, local_name(nid))


def label_looks_like_index(label: str) -> bool:
    """Heuristic: treat pure digits as non-human-readable index."""
    return bool(DIGITS_ONLY.match(label))


def extract_edges_and_node_ids(
    nodes: List[Dict[str, Any]],
    id2label: Dict[str, str],
    exclude_predicates: Optional[Set[str]] = None,
) -> Tuple[List[Tuple[str, str, str]], Set[str]]:
    if exclude_predicates is None:
        exclude_predicates = set(DEFAULT_EXCLUDE_PREDICATES)

    edges: List[Tuple[str, str, str]] = []
    used_node_ids: Set[str] = set()

    for node in nodes:
        sid = node.get("@id")
        if not isinstance(sid, str):
            continue

        for pid, obj in node.items():
            if not isinstance(pid, str) or pid.startswith("@"):
                continue
            if pid in exclude_predicates:
                continue

            objs = obj if isinstance(obj, list) else [obj]
            for o in objs:
                tid = is_entity_ref(o)
                if not tid:
                    continue

                s_label = readable_node_label(sid, id2label)
                p_label = readable_pred(pid, id2label)
                t_label = readable_node_label(tid, id2label)

                edges.append((s_label, p_label, t_label))
                used_node_ids.add(sid)
                used_node_ids.add(tid)

    return edges, used_node_ids


def write_edges_jsonl(edges: List[Tuple[str, str, str]], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        for s, p, t in edges:
            f.write(json.dumps({"source": s, "relation": p, "target": t}, ensure_ascii=False) + "\n")


def write_nodes_jsonl(
    nodes: List[Dict[str, Any]],
    id2label: Dict[str, str],
    edge_node_ids: Set[str],
    out_path: Path,
    drop_numeric_label_nodes: bool = True,
) -> None:
    """
    Writes nodes:
      - nodes present in @graph (source="graph")
      - plus nodes that only appear in edges but not in @graph (source="edge_only")

    Optimization:
      - If drop_numeric_label_nodes=True, we omit nodes whose label is pure digits.
    """
    node_records: Dict[str, Dict[str, Any]] = {}

    # nodes in graph
    for node in nodes:
        nid = node.get("@id")
        if not isinstance(nid, str):
            continue
        label = readable_node_label(nid, id2label)
        if drop_numeric_label_nodes and label_looks_like_index(label):
            continue

        node_records[nid] = {
            "id": nid,
            "label": label,
            "types": extract_types(node),
            "source": "graph",
        }

    # nodes only in edges (might not have labels)
    for nid in edge_node_ids:
        if nid in node_records:
            continue
        label = readable_node_label(nid, id2label)
        if drop_numeric_label_nodes and label_looks_like_index(label):
            continue

        node_records[nid] = {
            "id": nid,
            "label": label,
            "types": [],
            "source": "edge_only",
        }

    items = list(node_records.values())
    items.sort(key=lambda x: (x.get("label", ""), x.get("id", "")))

    with out_path.open("w", encoding="utf-8") as f:
        for rec in items:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> int:
    if len(sys.argv) < 4:
        print(
            "Usage: python jsonld_to_edges_and_nodes.py INPUT.jsonld OUTPUT_EDGES.jsonl OUTPUT_NODES.jsonl",
            file=sys.stderr,
        )
        return 2

    in_path = Path(sys.argv[1])
    edges_path = Path(sys.argv[2])
    nodes_path = Path(sys.argv[3])

    nodes = load_jsonld_nodes(in_path)
    id2label = build_id2label(nodes)
    edges, edge_node_ids = extract_edges_and_node_ids(nodes, id2label)

    write_edges_jsonl(edges, edges_path)
    write_nodes_jsonl(
        nodes,
        id2label,
        edge_node_ids,
        nodes_path,
        drop_numeric_label_nodes=True,
    )

    print(f"[ok] input: {in_path}")
    print(f"[ok] nodes_in_graph={len(nodes)} id2label={len(id2label)}")
    print(f"[ok] edges={len(edges)}")
    print(f"[ok] wrote edges: {edges_path}")
    print(f"[ok] wrote nodes (filtered): {nodes_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
