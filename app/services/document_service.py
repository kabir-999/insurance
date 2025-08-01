import requests
import PyPDF2
import docx
import io
from urllib.parse import urlparse

def download_document(url: str) -> bytes:
    """Downloads a document from a URL."""
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for bad status codes
    return response.content

def extract_text_from_pdf(pdf_content: bytes) -> str:
    """Extracts text from a PDF document."""
    text = ""
    pdf_file = io.BytesIO(pdf_content)
    reader = PyPDF2.PdfReader(pdf_file)
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def extract_text_from_docx(docx_content: bytes) -> str:
    """Extracts text from a DOCX document."""
    text = ""
    docx_file = io.BytesIO(docx_content)
    document = docx.Document(docx_file)
    for para in document.paragraphs:
        text += para.text + "\n"
    return text

def process_document(url: str) -> str:
    """Downloads and extracts text from a document based on its extension."""
    file_content = download_document(url)
    
    path = urlparse(url).path

    if path.lower().endswith('.pdf'):
        return extract_text_from_pdf(file_content)
    elif path.lower().endswith('.docx'):
        return extract_text_from_docx(file_content)
    # Add more document types here if needed
    else:
        # Assuming plain text for other types, or raise an error
        try:
            return file_content.decode('utf-8')
        except UnicodeDecodeError:
            raise ValueError("Unsupported file type or encoding")
