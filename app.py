import gradio as gr

from src.ingestion.upload_indexer import (
    create_uploaded_vector_store,
    create_uploaded_retriever,
)

from src.retrieval.retriever import retrieve_documents

from src.generation.generator import generate_lab


# ============================================================
# GLOBAL STATE
# ============================================================

current_vector_store = None
current_retriever = None


# ============================================================
# PROCESS TEXTBOOK
# ============================================================
def process_textbook(pdf_file):

    global current_vector_store
    global current_retriever

    if pdf_file is None:
        return "❌ Please upload a PDF textbook first."

    try:

        (
            current_vector_store,
            page_count,
            chunk_count,
            db_id,
        ) = create_uploaded_vector_store(
            pdf_file
        )

        current_retriever = create_uploaded_retriever(
            current_vector_store,
            k=8,
        )

        return (
            "✅ Textbook processed successfully!\n\n"
            "Your textbook is ready. "
            "You can now configure your programming lab."
        )

    except Exception as error:

        current_vector_store = None
        current_retriever = None

        return (
            "❌ Failed to process the textbook.\n\n"
            f"Error: {error}"
        )

# ============================================================
# GENERATE LAB
# ============================================================

def generate_lesson(
    course,
    topic,
    student_level,
    duration,
    requirements,
):

    global current_retriever

    if current_retriever is None:
        return (
            "❌ Please upload and process a textbook first."
        )

    if not topic.strip():
        return (
            "❌ Please enter a topic."
        )

    try:

        request = f"""
Create a programming laboratory based on the
uploaded textbook.

Course:
{course}

Topic:
{topic}

Student Level:
{student_level}

Duration:
{duration}

Additional Requirements:
{requirements}

Important:
The laboratory must respect the requested duration.
Do not add extra sessions.
"""

        documents = retrieve_documents(
            query=request,
            retriever=current_retriever,
        )

        if not documents:
            return (
                "❌ No relevant textbook content was found "
                "for this request."
            )

        lab = generate_lab(
            request,
            documents,
        )

        return format_lab(lab)

    except Exception as error:

        return (
            "❌ Failed to generate the laboratory.\n\n"
            f"Error: {error}"
        )


# ============================================================
# FORMAT GENERATED LAB
# ============================================================

def format_lab(lab):

    output = []

    output.append(
        f"# {lab.lab_title}"
    )

    output.append(
        f"**Course:** {lab.course}"
    )

    output.append(
        f"**Target Students:** {lab.target_students}"
    )

    output.append(
        f"**Duration:** {lab.duration}"
    )

    output.append(
        "\n## Learning Objectives"
    )

    for objective in lab.learning_objectives:
        output.append(
            f"- {objective}"
        )

    output.append(
        "\n## Introduction"
    )

    output.append(
        lab.introduction
    )

    output.append(
        "\n## Concepts"
    )

    for concept in lab.concepts:
        output.append(
            f"- {concept}"
        )

    output.append(
        "\n## Examples"
    )

    for example in lab.examples:

        output.append(
            f"### {example.title}"
        )

        output.append(
            example.explanation
        )

        output.append(
            "```python\n"
            + example.code
            + "\n```"
        )

        output.append(
            f"**Expected Output:** "
            f"{example.expected_output}"
        )

    output.append(
        "\n## Guided Practice"
    )

    for item in lab.guided_practice:
        output.append(
            f"- {item}"
        )

    output.append(
        "\n## Exercises"
    )

    for exercise in lab.exercises:

        output.append(
            f"### {exercise.title}"
        )

        output.append(
            exercise.instructions
        )

        output.append(
            f"**Difficulty:** "
            f"{exercise.difficulty}"
        )

    output.append(
        "\n## Homework"
    )

    for homework in lab.homework:
        output.append(
            f"- {homework}"
        )

    output.append(
        "\n## Summary"
    )

    output.append(
        lab.summary
    )

    return "\n".join(output)



# ============================================================
# GRADIO UI
# ============================================================

with gr.Blocks(
    title="TeachPlan-AI"
) as app:

    gr.Markdown(
        """
# 📚 TeachPlan-AI

### AI-Powered Programming Lab Generator

Generate structured programming labs grounded in
your uploaded programming textbook.
"""
    )

    with gr.Row():

        # ====================================================
        # LEFT COLUMN
        # ====================================================

        with gr.Column(scale=1):

            # ------------------------------------------------
            # TEXTBOOK
            # ------------------------------------------------

            gr.Markdown(
                "## 📖 Textbook"
            )

            pdf_file = gr.File(
                label="Programming Textbook",
                file_types=[".pdf"],
                type="filepath",
            )

            process_button = gr.Button(
                "📖 Process Textbook",
                variant="primary",
            )

            processing_status = gr.Markdown(
                "Upload a PDF and process it to continue."
            )

            process_button.click(
                fn=process_textbook,
                inputs=pdf_file,
                outputs=processing_status,
            )

            # ------------------------------------------------
            # LAB INFORMATION
            # ------------------------------------------------

            gr.Markdown(
                "## ⚙️ Lab Information"
            )

            course = gr.Textbox(
                label="Course",
                value="Elementary Programming",
                placeholder="e.g. Elementary Programming",
            )

            topic = gr.Textbox(
                label="Topic",
                placeholder="e.g. Introduction to Python",
            )

            student_level = gr.Dropdown(
                label="Student Level",
                choices=[
                    "Beginner",
                    "Intermediate",
                    "Advanced",
                ],
                value="Beginner",
            )

            duration = gr.Textbox(
                label="Duration",
                value="90 minutes",
                placeholder="e.g. 90 minutes",
            )

            requirements = gr.Textbox(
                label="Additional Requirements",
                placeholder=(
                    "e.g. Include simple examples "
                    "and beginner exercises."
                ),
                lines=4,
            )

            generate_button = gr.Button(
                "✨ Generate Programming Lab",
                variant="primary",
            )

        # ====================================================
        # RIGHT COLUMN
        # ====================================================

        with gr.Column(
            scale=2
        ):

            gr.Markdown(
                "## 📝 Generated Lab"
            )

            generated_lab = gr.Markdown(
                value=(
                    "Your generated programming lab "
                    "will appear here."
                )
            )

            generate_button.click(
                fn=generate_lesson,
                inputs=[
                    course,
                    topic,
                    student_level,
                    duration,
                    requirements,
                ],
                outputs=generated_lab,
            )


# ============================================================
# RUN APP
# ============================================================

if __name__ == "__main__":

    app.launch()
