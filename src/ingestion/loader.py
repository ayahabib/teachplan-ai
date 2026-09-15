from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader


def load_pdf(pdf_path: str | Path):
    """
    Load a PDF file and return its pages
    as LangChain Documents.
    """

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(
            "Only PDF files are supported."
        )

    loader = PyPDFLoader(str(pdf_path))

    documents = loader.load()

    return documents