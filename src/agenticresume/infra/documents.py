from pathlib import Path
from pypdf import PdfReader

def _read_pdf(path: Path)-> str:
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def read_document(path: Path | str) -> str:
    #reads a pdf, txt or md file
    
    path = Path(path)
    match path.suffix.lower():
        case ".pdf":
            return _read_pdf(path)
        case ".txt" | ".md":
            return path.read_text(encoding="utf-8")
        case other:
            raise ValueError(f"unsupported document type: {other!r}")

    