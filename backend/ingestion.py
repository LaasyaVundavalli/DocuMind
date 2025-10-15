import os
import logging
from typing import List, Dict
# Removed PyMuPDF (fitz) import since it requires Visual Studio build tools
from PyPDF2 import PdfReader
from docx import Document
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract text content from a PDF file using PyPDF2.

    Args:
        file_path (str): Path to the PDF file.

    Returns:
        str: Extracted text content from the PDF.

    Raises:
        Exception: If there's an error reading the PDF file.

    Example:
        >>> text = extract_text_from_pdf("document.pdf")
        >>> print(text[:100])
    """
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n"
            
        logger.info(f"Successfully extracted text from PDF using PyPDF2: {file_path}")
        return text.strip()
            
    except Exception as e:
        logger.error(f"Error extracting text from PDF {file_path}: {str(e)}")
        raise Exception(f"Failed to extract text from PDF: {str(e)}")


def extract_text_from_docx(file_path: str) -> str:
    """
    Extract text content from a DOCX file.

    Args:
        file_path (str): Path to the DOCX file.

    Returns:
        str: Extracted text content from the DOCX file.

    Raises:
        Exception: If there's an error reading the DOCX file.

    Example:
        >>> text = extract_text_from_docx("document.docx")
        >>> print(text[:100])
    """
    try:
        doc = Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        logger.info(f"Successfully extracted text from DOCX: {file_path}")
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting text from DOCX {file_path}: {str(e)}")
        raise Exception(f"Failed to extract text from DOCX: {str(e)}")


def extract_text_from_txt(file_path: str) -> str:
    """
    Extract text content from a TXT file.

    Args:
        file_path (str): Path to the TXT file.

    Returns:
        str: Extracted text content from the TXT file.

    Raises:
        Exception: If there's an error reading the TXT file.

    Example:
        >>> text = extract_text_from_txt("document.txt")
        >>> print(text[:100])
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        logger.info(f"Successfully extracted text from TXT: {file_path}")
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting text from TXT {file_path}: {str(e)}")
        raise Exception(f"Failed to extract text from TXT: {str(e)}")


def ingest_documents(file_list: List[str]) -> List[Dict[str, str]]:
    """
    Ingest multiple documents and extract their text content.

    This function processes a list of file paths, determines the file type based on extension,
    and extracts text accordingly. It handles errors gracefully for unsupported or corrupted files.

    Args:
        file_list (List[str]): List of file paths to process.

    Returns:
        List[Dict[str, str]]: List of dictionaries containing 'file_name' and 'content' for each successfully processed file.

    Example:
        >>> files = ["doc1.pdf", "doc2.docx", "doc3.txt"]
        >>> documents = ingest_documents(files)
        >>> print(f"Processed {len(documents)} documents")
    """
    documents = []
    supported_extensions = {'.pdf', '.docx', '.txt'}

    for file_path in file_list:
        try:
            if not os.path.exists(file_path):
                logger.warning(f"File does not exist: {file_path}")
                continue

            file_name = os.path.basename(file_path)
            file_extension = os.path.splitext(file_path)[1].lower()

            if file_extension not in supported_extensions:
                logger.warning(f"Unsupported file type: {file_extension} for file {file_name}")
                continue

            if file_extension == '.pdf':
                content = extract_text_from_pdf(file_path)
            elif file_extension == '.docx':
                content = extract_text_from_docx(file_path)
            elif file_extension == '.txt':
                content = extract_text_from_txt(file_path)

            documents.append({
                "file_name": file_name,
                "content": content
            })

            logger.info(f"Successfully ingested document: {file_name}")

        except Exception as e:
            logger.error(f"Failed to ingest document {file_path}: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            # Continue processing other files even if one fails

    logger.info(f"Total documents ingested: {len(documents)} out of {len(file_list)}")
    return documents


# Example usage
if __name__ == "__main__":
    # Example file paths (replace with actual paths)
    example_files = [
        "data/uploaded_docs/sample.pdf",
        "data/uploaded_docs/sample.docx",
        "data/uploaded_docs/sample.txt"
    ]

    # Ingest documents
    ingested_docs = ingest_documents(example_files)

    # Print results
    for doc in ingested_docs:
        print(f"File: {doc['file_name']}")
        print(f"Content preview: {doc['content'][:200]}...")
        print("-" * 50)