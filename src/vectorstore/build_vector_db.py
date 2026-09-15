import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv


load_dotenv()


CHUNKS_PATH = Path("data/processed/chunks.json")
VECTOR_DB_PATH = Path("data/vector_db")


def load_chunks():
    with open(
        CHUNKS_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    documents = []

    for item in data:
        document = Document(
            page_content=item["page_content"],
            metadata=item["metadata"],
        )

        # Skip empty or invalid chunks
        if not document.page_content.strip():
            continue

        if (
            document.metadata["page_start"]
            > document.metadata["page_end"]
        ):
            continue

        documents.append(document)

    return documents


def build_vector_db(documents):
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large"
    )

    vector_store = Chroma(
        collection_name="teachplan",
        embedding_function=embeddings,
        persist_directory=str(VECTOR_DB_PATH),
    )

    vector_store.add_documents(documents)

    return vector_store


if __name__ == "__main__":
    documents = load_chunks()

    print(f"Loaded documents: {len(documents)}")

    vector_store = build_vector_db(documents)

    print("\n==============================")
    print("Vector database created!")
    print("==============================")