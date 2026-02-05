"""
Shared utility helpers for the InterfaceML web app.
"""

from __future__ import annotations

from flask import current_app


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    allowed = current_app.config.get('ALLOWED_EXTENSIONS', set())
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed


def compress_ranges(indices: list[int]) -> list[tuple[int, int]]:
    """Compress sorted indices into inclusive ranges."""
    if not indices:
        return []

    sorted_idx = sorted(set(indices))
    ranges: list[tuple[int, int]] = []
    start = sorted_idx[0]
    prev = sorted_idx[0]

    for value in sorted_idx[1:]:
        if value == prev + 1:
            prev = value
            continue
        ranges.append((start, prev))
        start = value
        prev = value

    ranges.append((start, prev))
    return ranges


def format_ranges(ranges: list[tuple[int, int]]) -> str:
    """Format ranges for CP2K LIST syntax (e.g., 1..12 14 18..25)."""
    parts = []
    for start, end in ranges:
        if start == end:
            parts.append(f"{start}")
        else:
            parts.append(f"{start}..{end}")
    return " ".join(parts)
