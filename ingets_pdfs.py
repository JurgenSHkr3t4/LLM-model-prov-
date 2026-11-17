"""Copy PDFs into the ingest folder and print a summary."""
import shutil
import sys
from pathlib import Path

from src import config

def ingest(paths: list[str]) -> None:
    config.PDF_DIR.mkdir(parents=True, exist_ok=True)
    for p in paths:
        src = Path(p)
        if not src.exists():
            print(f"⏭  Skipped (not found): {src}")
            continue
        if src.suffix.lower() != ".pdf":
            print(f"⏭  Skipped (not a PDF): {src}")
            continue
        dest = config.PDF_DIR / src.name
        shutil.copy2(src, dest)
        print(f"✅ {src.name} → {dest}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/ingest_pdfs.py <file1.pdf> [file2.pdf ...]")
        sys.exit(1)
    ingest(sys.argv[1:])
