"""Utilities for hierarchical code paths."""

from __future__ import annotations


def code_to_path(code: str, delim: str = " > ") -> list[str]:
    """Split a hierarchical code string into a list of segments."""
    return [segment.strip() for segment in code.split(delim)]


def build_code_tree(codes: list[str], delim: str = " > ") -> dict:
    """Build a nested dictionary representing the hierarchy of codes."""
    tree: dict[str, dict] = {}
    for code in codes:
        path = code_to_path(code, delim=delim)
        node = tree
        for segment in path:
            node = node.setdefault(segment, {})
    return tree


def common_ancestor_depth(a: str, b: str, delim: str = " > ") -> int:
    """Return the number of matching path segments from the root."""
    path_a = code_to_path(a, delim=delim)
    path_b = code_to_path(b, delim=delim)
    depth = 0
    for seg_a, seg_b in zip(path_a, path_b):
        if seg_a != seg_b:
            break
        depth += 1
    return depth
