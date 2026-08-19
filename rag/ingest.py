"""
Turns raw uploaded files into (text, metadata) chunks ready to embed.
Supports PDF, DOCX, TXT, Markdown and images (via captioning).
"""
import io
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import docx
import markdown as md_lib
import pdfplumber
from bs4 import BeautifulSoup

from rag import config, vision

logger = logging.getLogger(__name__)

SUPPORTED_TEXT_EXT = {".pdf", ".docx", ".txt", ".md"}
SUPPORTED_IMAGE_EXT = {".png", ".jpg", ".jpeg"}
SUPPORTED_EXT = SUPPORTED_TEXT_EXT | SUPPORTED_IMAGE_EXT


def extract_text(filename: str, file_bytes: bytes) -> str:
    """Extract raw text from a supported file type."""
    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        text_parts = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text_parts.append(page_text)
        return "\n".join(text_parts)

    if ext == ".docx":
        document = docx.Document(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in document.paragraphs)

    if ext == ".txt":
        return file_bytes.decode("utf-8", errors="ignore")

    if ext == ".md":
        html = md_lib.markdown(file_bytes.decode("utf-8", errors="ignore"))
        return BeautifulSoup(html, "html.parser").get_text()

    if ext in SUPPORTED_IMAGE_EXT:
        return vision.caption_image(file_bytes)

    raise ValueError(
        f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXT)}"
    )


def chunk_text(
    text: str,
    chunk_size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
) -> List[str]:
    """Simple sliding-window chunker over whitespace-normalized text."""
    text = " ".join(text.split())
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def process_file(filename: str, file_bytes: bytes) -> Tuple[List[str], List[Dict]]:
    """
    Returns (chunks, metadatas) ready to hand to the embedder + vector store.
    Images produce a single "chunk" (the caption) rather than being split.
    """
    ext = Path(filename).suffix.lower()
    text = extract_text(filename, file_bytes)

    if ext in SUPPORTED_IMAGE_EXT:
        chunks = [text] if text else []
    else:
        chunks = chunk_text(text)

    metadatas = [
        {"source": filename, "chunk_index": i, "file_type": ext.lstrip(".")}
        for i in range(len(chunks))
    ]
    return chunks, metadatas
