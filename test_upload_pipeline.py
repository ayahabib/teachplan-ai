from pathlib import Path

from src.ingestion.loader import load_pdf
from src.chunking.llm_chunker import process_in_batches


PDF_PATH = Path(
    "data/raw/Pages From Y_Daniel_Liang_Introduction_to_Programmi.pdf"
)


def main():

    print("Loading PDF...")

    documents = load_pdf(PDF_PATH)

    print(f"Total pages loaded: {len(documents)}")

    # Test only first 3 pages
    test_documents = documents[66:73]

    print(
        f"Testing with {len(test_documents)} pages..."
    )

    chunks = process_in_batches(
        test_documents,
        source=PDF_PATH.name,
        batch_size=3,
        overlap=0,
        max_workers=1,
    )

    print(
        f"\nCreated chunks: {len(chunks)}"
    )

    print("\n==============================")
    print("CHUNK RESULTS")
    print("==============================")

    for index, chunk in enumerate(
        chunks,
        start=1,
    ):

        print(
            f"\nChunk {index}"
        )

        print(
            "Type:",
            chunk.metadata.get("type"),
        )

        print(
            "Section:",
            chunk.metadata.get("section"),
        )

        print(
            "Title:",
            chunk.metadata.get("title"),
        )

        print(
            "Pages:",
            chunk.metadata.get("page_start"),
            "-",
            chunk.metadata.get("page_end"),
        )

        print(
            "Characters:",
            len(chunk.page_content),
        )

        print(
            "Preview:",
            chunk.page_content[:300]
            .replace("\n", " "),
        )

    print("\n==============================")
    print("TEST COMPLETED")
    print("==============================")


if __name__ == "__main__":
    main()