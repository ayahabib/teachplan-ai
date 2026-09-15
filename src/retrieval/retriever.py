from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


load_dotenv()


VECTOR_DB_PATH = Path(
    "data/vector_db"
)


def create_retriever(k=10):

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large"
    )

    vector_store = Chroma(
        collection_name="teachplan",
        embedding_function=embeddings,
        persist_directory=str(
            VECTOR_DB_PATH
        ),
    )

    return vector_store.as_retriever(
        search_kwargs={
            "k": k
        }
    )


def retrieve_documents(
    query,
    retriever=None,
):

    if retriever is None:
        retriever = create_retriever()

    documents = retriever.invoke(
        query
    )

    filtered_documents = []

    for document in documents:

        content = (
            document.page_content.strip()
        )

        if len(content) < 150:
            continue

        if (
            "Chapter 2 Elementary Programming"
            in content
            and len(content) < 500
        ):
            continue

        filtered_documents.append(
            document
        )

    return filtered_documents