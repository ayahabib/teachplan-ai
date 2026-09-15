import shutil
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from src.ingestion.loader import load_pdf
from src.chunking.llm_chunker import process_in_batches


load_dotenv()


# ============================================================
# PATHS
# ============================================================

UPLOAD_DB_ROOT = Path(
    "data/uploaded_vector_dbs"
)


# ============================================================
# CREATE VECTOR STORE
# ============================================================

def create_uploaded_vector_store(pdf_path: str):

    """
    Load an uploaded PDF, perform LLM-assisted chunking,
    create embeddings, and store the chunks in a unique
    Chroma database.
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

    # --------------------------------------------------------
    # Create unique database for this upload
    # --------------------------------------------------------

    db_id = uuid.uuid4().hex

    db_path = (
        UPLOAD_DB_ROOT / db_id
    )

    db_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:

        # ----------------------------------------------------
        # Load PDF
        # ----------------------------------------------------

        documents = load_pdf(
            pdf_path
        )

        if not documents:
            raise ValueError(
                "No pages were found in the PDF."
            )

        # ----------------------------------------------------
        # LLM-assisted chunking
        # ----------------------------------------------------

        chunks = process_in_batches(
            documents,
            source=pdf_path.name,
        )

        # ----------------------------------------------------
        # Remove empty chunks
        # ----------------------------------------------------

        chunks = [
            document
            for document in chunks
            if document.page_content.strip()
        ]

        if not chunks:
            raise ValueError(
                "No usable chunks were created from the PDF."
            )

        # ----------------------------------------------------
        # Embeddings
        # ----------------------------------------------------

        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large"
        )

        # ----------------------------------------------------
        # Chroma
        # ----------------------------------------------------

        vector_store = Chroma(
            collection_name=f"upload_{db_id}",
            embedding_function=embeddings,
            persist_directory=str(db_path),
        )

        vector_store.add_documents(
            chunks
        )

        return (
            vector_store,
            len(documents),
            len(chunks),
            db_id,
        )

    except Exception:

        # Remove partially created DB
        shutil.rmtree(
            db_path,
            ignore_errors=True,
        )

        raise


# ============================================================
# CREATE RETRIEVER
# ============================================================

def create_uploaded_retriever(
    vector_store,
    k=8,
):

    """
    Create a retriever from the uploaded PDF
    vector store.
    """

    return vector_store.as_retriever(
        search_kwargs={
            "k": k
        }
    )