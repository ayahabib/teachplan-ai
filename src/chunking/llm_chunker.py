from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum

from dotenv import load_dotenv
from langchain_core.documents import Document
from openai import OpenAI
from pydantic import BaseModel
from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "gpt-5-nano"

BATCH_SIZE = 30
BATCH_OVERLAP = 1
MAX_WORKERS = 4


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()

client = OpenAI()


# ============================================================
# DATA MODELS
# ============================================================

class SegmentType(str, Enum):
    SECTION = "section"
    SUBSECTION = "subsection"
    EXERCISE = "exercise"
    REVIEW_QUESTIONS = "review_questions"
    EXAMPLE = "example"


class Segment(BaseModel):
    type: SegmentType
    number: str | None = None
    title: str | None = None
    start_line: int
    end_line: int | None = None


class DocumentStructure(BaseModel):
    segments: list[Segment]


# ============================================================
# LLM STRUCTURE ANALYSIS
# ============================================================

def analyze_structure(text: str):
    response = client.responses.parse(
        model=MODEL_NAME,
        input=f"""
Analyze the structure of the following textbook text.

The text is numbered by line.

Your task is to identify meaningful educational sections
that can be used as retrieval chunks for a RAG system.

Rules:

- A heading with numbering like 1.1, 1.2, 2.1 is a section.
- A heading with numbering like 1.2.1, 1.2.2, 2.1.1 is a subsection.
- "Review Questions" should be classified as review_questions.
- Exercises should be classified as exercise.
- Examples should be classified as example.
- start_line must be the line number where the segment starts.
- Do not rewrite the textbook text.
- Only identify segments that actually appear in the text.
- Do not invent sections.
- Keep the original order of the segments.
- Do not create segments for page numbers, table-of-contents entries,
  isolated questions, or fragments that are not meaningful sections.
- Prefer meaningful content sections over tiny fragments.

Important:

The first pages of the textbook may contain a table of contents.
Do not treat table-of-contents entries as actual content sections
unless they clearly contain the actual section content.

Text:

{text}
""",
        text_format=DocumentStructure,
    )

    return response.output_parsed


# ============================================================
# NUMBERED LINES
# ============================================================

def build_numbered_lines(documents):
    lines = []
    line_pages = []

    for document in documents:

        page_number = (
            document.metadata.get("page", 0) + 1
        )

        for line in document.page_content.splitlines():

            if line.strip():

                lines.append(line)
                line_pages.append(page_number)

    return lines, line_pages


# ============================================================
# ADD END LINES
# ============================================================

def add_end_lines(segments, total_lines):

    valid_segments = [
        segment
        for segment in segments
        if 1 <= segment.start_line <= total_lines
    ]

    valid_segments.sort(
        key=lambda segment: segment.start_line
    )

    cleaned_segments = []

    for index, segment in enumerate(valid_segments):

        if index < len(valid_segments) - 1:

            next_start = (
                valid_segments[index + 1].start_line
            )

            segment.end_line = next_start - 1

        else:

            segment.end_line = total_lines

        if segment.end_line < segment.start_line:
            continue

        cleaned_segments.append(segment)

    return cleaned_segments


# ============================================================
# EXTRACT SEGMENT TEXT
# ============================================================

def extract_segment_text(lines, segment):

    start = segment.start_line - 1
    end = segment.end_line

    return "\n".join(
        lines[start:end]
    ).strip()


# ============================================================
# CONVERT SEGMENT TO LANGCHAIN DOCUMENT
# ============================================================

def segment_to_document(
    lines,
    line_pages,
    segment,
    source,
):

    text = extract_segment_text(
        lines,
        segment,
    )

    start_page = line_pages[
        segment.start_line - 1
    ]

    end_page = line_pages[
        segment.end_line - 1
    ]

    return Document(
        page_content=text,
        metadata={
            "type": segment.type.value,
            "section": segment.number,
            "title": segment.title,
            "page_start": start_page,
            "page_end": end_page,
            "source": source,
        },
    )


# ============================================================
# CONVERT ALL SEGMENTS
# ============================================================

def segments_to_documents(
    lines,
    line_pages,
    segments,
    source,
):

    documents = []

    for segment in segments:

        if segment.start_line is None:
            continue

        if segment.end_line is None:
            continue

        if segment.start_line < 1:
            continue

        if segment.end_line > len(lines):
            continue

        if segment.start_line > segment.end_line:
            continue

        document = segment_to_document(
            lines,
            line_pages,
            segment,
            source,
        )

        if not document.page_content.strip():
            continue

        documents.append(document)

    return documents


# ============================================================
# PROCESS ONE DOCUMENT BATCH
# ============================================================

def process_documents(documents, source="uploaded.pdf"):

    lines, line_pages = build_numbered_lines(
        documents
    )

    if not lines:
        return []

    numbered_text = "\n".join(
        f"{index}: {line}"
        for index, line in enumerate(
            lines,
            start=1,
        )
    )

    structure = analyze_structure(
        numbered_text
    )

    segments = add_end_lines(
        structure.segments,
        len(lines),
    )

    return segments_to_documents(
        lines,
        line_pages,
        segments,
        source,
    )


# ============================================================
# PROCESS ONE BATCH
# ============================================================

def process_one_batch(
    documents,
    start,
    batch_size,
    overlap,
    source,
):

    core_end = min(
        start + batch_size,
        len(documents),
    )

    batch_start = max(
        0,
        start - overlap,
    )

    batch = documents[
        batch_start:core_end
    ]

    return process_documents(
        batch,
        source=source,
    )


# ============================================================
# PROCESS BATCHES IN PARALLEL
# ============================================================

def process_in_batches(
    documents,
    source="uploaded.pdf",
    batch_size=BATCH_SIZE,
    overlap=BATCH_OVERLAP,
    max_workers=MAX_WORKERS,
):

    starts = list(
        range(
            0,
            len(documents),
            batch_size,
        )
    )

    total_batches = len(starts)

    all_documents = []

    with ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:

        futures = {
            executor.submit(
                process_one_batch,
                documents,
                start,
                batch_size,
                overlap,
                source,
            ): index
            for index, start in enumerate(starts)
        }

        results = {}

        for future in tqdm(
            as_completed(futures),
            total=total_batches,
            desc="Processing LLM batches",
            unit="batch",
        ):

            index = futures[future]

            results[index] = future.result()

    for index in sorted(results):

        all_documents.extend(
            results[index]
        )

    return deduplicate_documents(
        all_documents
    )


# ============================================================
# DEDUPLICATION
# ============================================================

def deduplicate_documents(documents):

    grouped = {}

    for document in documents:

        key = (
            document.metadata.get("type"),
            document.metadata.get("section"),
            document.metadata.get("title"),
            document.metadata.get("page_start"),
        )

        if key not in grouped:

            grouped[key] = document

        elif len(document.page_content) > len(
            grouped[key].page_content
        ):

            grouped[key] = document

    return list(
        grouped.values()
    )


# ============================================================
# VALIDATION
# ============================================================

def validate_documents(documents):

    print("\n==============================")
    print("DOCUMENT VALIDATION")
    print("==============================")

    print(
        f"Total Documents: {len(documents)}"
    )

    empty_documents = [
        document
        for document in documents
        if not document.page_content.strip()
    ]

    print(
        f"Empty Documents: "
        f"{len(empty_documents)}"
    )

    required_metadata = {
        "type",
        "section",
        "title",
        "page_start",
        "page_end",
        "source",
    }

    missing_metadata = []

    for index, document in enumerate(
        documents,
        start=1,
    ):

        missing = (
            required_metadata
            - set(document.metadata.keys())
        )

        if missing:
            missing_metadata.append(
                (index, missing)
            )

    print(
        f"Documents with Missing Metadata: "
        f"{len(missing_metadata)}"
    )

    invalid_page_ranges = []

    for index, document in enumerate(
        documents,
        start=1,
    ):

        start_page = document.metadata[
            "page_start"
        ]

        end_page = document.metadata[
            "page_end"
        ]

        if start_page > end_page:
            invalid_page_ranges.append(index)

    print(
        f"Invalid Page Ranges: "
        f"{len(invalid_page_ranges)}"
    )

    character_lengths = [
        len(document.page_content)
        for document in documents
    ]

    if character_lengths:

        print(
            f"Shortest Document: "
            f"{min(character_lengths)} chars"
        )

        print(
            f"Longest Document: "
            f"{max(character_lengths)} chars"
        )

        print(
            f"Average Document: "
            f"{sum(character_lengths) // len(character_lengths)} chars"
        )

    print("\n--- Sample Documents ---")

    for index, document in enumerate(
        documents[:5],
        start=1,
    ):

        print(
            f"\nDocument {index}"
        )

        print(
            "Type:",
            document.metadata["type"],
        )

        print(
            "Section:",
            document.metadata["section"],
        )

        print(
            "Title:",
            document.metadata["title"],
        )

        print(
            "Pages:",
            document.metadata["page_start"],
            "-",
            document.metadata["page_end"],
        )

        print(
            "Characters:",
            len(document.page_content),
        )

        print(
            "Preview:",
            document.page_content[:300]
            .replace("\n", " "),
        )

    print("\n==============================")

    if (
        len(empty_documents) == 0
        and len(missing_metadata) == 0
        and len(invalid_page_ranges) == 0
    ):

        print(
            "Validation Status: PASSED"
        )

    else:

        print(
            "Validation Status: FAILED"
        )

    print(
        "=============================="
    )