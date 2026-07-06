"""Document readers for ingesting files and folders into the RAG store.

Extracts plain text from ``.txt``, ``.md``/``.markdown`` and ``.pdf`` files and
provides helpers to walk a directory. Keeping this separate from
:mod:`interview_helper.data_loader` lets the loader stay focused on chunking
while readers focus on getting text out of different file formats.
"""

import logging
from pathlib import Path
from typing import Iterator, List, Optional, Set, Tuple, Union


logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

TEXT_EXTENSIONS: Set[str] = {".txt", ".md", ".markdown", ".text"}
PDF_EXTENSIONS: Set[str] = {".pdf"}
SUPPORTED_EXTENSIONS: Set[str] = TEXT_EXTENSIONS | PDF_EXTENSIONS


def is_supported(path: PathLike) -> bool:
    """Return True if the file extension is a supported document type."""
    return Path(path).suffix.lower() in SUPPORTED_EXTENSIONS


def extract_text(path: PathLike) -> str:
    """Extract plain text from a single document.

    Args:
        path: Path to a ``.txt``, ``.md`` or ``.pdf`` file.

    Returns:
        The extracted text.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the file type is not supported.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()
    if ext in PDF_EXTENSIONS:
        return _read_pdf(path)
    if ext in TEXT_EXTENSIONS:
        return path.read_text(encoding="utf-8", errors="ignore")

    raise ValueError(
        f"Unsupported file type '{ext}'. Supported: "
        f"{', '.join(sorted(SUPPORTED_EXTENSIONS))}"
    )


def _read_pdf(path: Path) -> str:
    """Extract text from a PDF using pypdf."""
    try:
        from pypdf import PdfReader
    except ImportError as e:  # pragma: no cover - dependency guard
        raise ImportError(
            "Reading PDFs requires 'pypdf'. Install it with: pip install pypdf"
        ) from e

    reader = PdfReader(str(path))
    pages = [(page.extract_text() or "") for page in reader.pages]
    text = "\n".join(pages).strip()
    if not text:
        logger.warning(f"No extractable text found in PDF (scanned image?): {path}")
    return text


def discover_files(
    directory: PathLike,
    extensions: Optional[Set[str]] = None,
    recursive: bool = True,
) -> List[Path]:
    """Find supported document files within a directory.

    Args:
        directory: Folder to scan.
        extensions: Set of lowercase extensions to include (defaults to all
            supported types).
        recursive: Whether to descend into subdirectories.

    Returns:
        A sorted list of matching file paths.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    exts = {e.lower() for e in (extensions or SUPPORTED_EXTENSIONS)}
    globber = directory.rglob("*") if recursive else directory.glob("*")
    return sorted(
        p for p in globber if p.is_file() and p.suffix.lower() in exts
    )


def iter_documents(path: PathLike) -> Iterator[Tuple[Path, str]]:
    """Yield ``(file_path, text)`` for a file or every file in a directory.

    Files that fail to read (unsupported, corrupt, empty) are skipped with a
    warning so one bad file does not abort a whole folder ingest.
    """
    path = Path(path)
    if path.is_dir():
        for file_path in discover_files(path):
            try:
                yield file_path, extract_text(file_path)
            except Exception as e:
                logger.warning(f"Skipping {file_path}: {e}")
    else:
        yield path, extract_text(path)
