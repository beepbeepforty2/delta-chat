"""FormatAdapter: one interface, N formats. New formats plug in here."""
from __future__ import annotations
import hashlib
from abc import ABC, abstractmethod
from src.canonical.model import BBox, CanonicalDocument


def element_id(sheet: int, etype: str, content: str, bbox: BBox) -> str:
    """Content+bbox hash, stable across re-ingestion runs -- shared by
    every FormatAdapter so citation ids don't shift between formats or
    between runs on the same source."""
    key = f"{sheet}|{etype}|{content}|{bbox.x0:.4f},{bbox.y0:.4f},{bbox.x1:.4f},{bbox.y1:.4f}"
    return "el_" + hashlib.sha1(key.encode()).hexdigest()[:12]


class FormatAdapter(ABC):
    format_name: str

    @abstractmethod
    def detect(self, path: str) -> bool:
        """Cheap sniff: is this file mine?"""

    @abstractmethod
    def ingest(self, pid: str, path: str) -> CanonicalDocument:
        """Resolve bytes -> canonical representation. Must populate
        extraction_confidence and retain rasters for markup."""

def resolve(pid_path: str, adapters: list[FormatAdapter]) -> CanonicalDocument:
    for a in adapters:
        if a.detect(pid_path):
            return a.ingest(pid_path, pid_path)
    raise ValueError(f"no adapter claims {pid_path}")
