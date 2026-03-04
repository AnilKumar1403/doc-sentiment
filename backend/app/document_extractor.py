from io import BytesIO
from pathlib import Path

import fitz
import pytesseract
from PIL import Image
from docx import Document as DocxDocument
from pypdf import PdfReader

from .config import get_settings

SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx", ".png", ".jpg", ".jpeg", ".tiff"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff"}

settings = get_settings()
pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd


class ExtractionError(Exception):
    pass


def extract_text_from_upload(filename: str, content: bytes) -> tuple[str, str]:
    suffix = Path(filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ExtractionError(
            "Unsupported file type. Supported: .txt, .pdf, .docx, .png, .jpg, .jpeg, .tiff"
        )

    if suffix == ".txt":
        return _extract_txt(content), "file"
    if suffix == ".docx":
        return _extract_docx(content), "file"
    if suffix == ".pdf":
        return _extract_pdf(content), "file"
    if suffix in IMAGE_EXTENSIONS:
        return _extract_image_ocr(content), "ocr"

    raise ExtractionError("Unsupported file type")


def _extract_txt(content: bytes) -> str:
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return content.decode("latin-1", errors="ignore")


def _extract_docx(content: bytes) -> str:
    doc = DocxDocument(BytesIO(content))
    return "\n".join(p.text.strip() for p in doc.paragraphs if p.text and p.text.strip())


def _extract_pdf(content: bytes) -> str:
    reader = PdfReader(BytesIO(content))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n".join(pages).strip()
    if text:
        return text

    # Fallback for scanned PDFs: render page images and OCR them.
    try:
        with fitz.open(stream=content, filetype="pdf") as doc:
            ocr_pages = []
            for page in doc:
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img = Image.open(BytesIO(pix.tobytes("png")))
                ocr_pages.append(pytesseract.image_to_string(img))
            return "\n".join(ocr_pages).strip()
    except pytesseract.TesseractNotFoundError as exc:
        raise ExtractionError(
            "Tesseract is not installed. Install Tesseract OCR to process scanned PDFs/images."
        ) from exc


def _extract_image_ocr(content: bytes) -> str:
    try:
        image = Image.open(BytesIO(content))
        return pytesseract.image_to_string(image).strip()
    except pytesseract.TesseractNotFoundError as exc:
        raise ExtractionError(
            "Tesseract is not installed. Install Tesseract OCR to analyze image documents."
        ) from exc
