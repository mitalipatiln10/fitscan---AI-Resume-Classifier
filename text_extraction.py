"""
Text extraction utilities — pulled directly from the notebook pipeline.
Supports .docx, .pdf, and legacy .doc (via LibreOffice conversion).
"""

import os
import docx
import pdfplumber


def extract_text_from_docx(file_path: str) -> str:
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])


def extract_text_from_pdf(file_path: str) -> str:
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def extract_text_from_doc(file_path: str, temp_folder: str = "temp_docx") -> str:
    """
    Legacy .doc support via LibreOffice headless conversion.
    Requires LibreOffice installed on the server. Adjust soffice_path per OS:
      - Windows: r"C:\\Program Files\\LibreOffice\\program\\soffice.exe"
      - Linux/Mac: usually just "soffice" (must be on PATH)
    """
    import subprocess

    os.makedirs(temp_folder, exist_ok=True)
    soffice_path = "soffice"  # change if needed for your deployment OS

    subprocess.run(
        [soffice_path, "--headless", "--convert-to", "docx", "--outdir", temp_folder, file_path],
        check=True,
    )

    converted_filename = os.path.splitext(os.path.basename(file_path))[0] + ".docx"
    converted_path = os.path.join(temp_folder, converted_filename)
    return extract_text_from_docx(converted_path)


def extract_text_any(file_path: str) -> str:
    """Dispatch based on file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".docx":
        return extract_text_from_docx(file_path)
    elif ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext == ".doc":
        return extract_text_from_doc(file_path)
    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
