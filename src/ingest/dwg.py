"""DWG adapter: REAL STUB behind the real seam.

The seam is honored: detect() works, ingest() documents the exact path and
raises. Implementation path (documented, not hypothetical):
  1. DWG -> DXF via ODA File Converter (free CLI) or LibreDWG `dwg2dxf`
  2. DXF -> canonical via ezdxf: text/mtext/dimension entities carry insert
     points and layer; geometry entities map to type="geometry"
  3. extraction_confidence = 1.0 (vector source)
Cut rationale in README: OCR path exercises more Applied-AI judgment for
the same time budget; the DXF leg of our own generator proves the entity
model is compatible.
"""
from src.ingest.base import FormatAdapter
from src.canonical.model import CanonicalDocument

class DWGAdapter(FormatAdapter):
    format_name = "dwg"

    def detect(self, path: str) -> bool:
        if not path.lower().endswith(".dwg"):
            return False
        with open(path, "rb") as f:
            return f.read(2) == b"AC"

    def ingest(self, pid: str, path: str) -> CanonicalDocument:
        raise NotImplementedError(
            "DWG ingestion stubbed: convert with ODA/LibreDWG to DXF, then "
            "parse with ezdxf (see module docstring). The adapter seam is real; "
            "see src/ingest/pdf_native.py for a complete adapter.")
