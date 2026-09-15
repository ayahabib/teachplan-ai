from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from src.retrieval.retriever import create_retriever
from src.retrieval.retriever import retrieve_documents

load_dotenv()

client = OpenAI()


class Example(BaseModel):
    title: str
    code: str
    explanation: str
    expected_output: str


class Exercise(BaseModel):
    title: str
    instructions: str
    difficulty: str


class ProgrammingLab(BaseModel):
    lab_title: str
    course: str
    target_students: str
    duration: str
    learning_objectives: list[str]

    introduction: str
    concepts: list[str]

    examples: list[Example]

    guided_practice: list[str]
    exercises: list[Exercise]

    homework: list[str]
    summary: str


def generate_lab(request, documents):

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    prompt = f"""
You are an educational content generator
for an Elementary Programming course.

Create a complete programming laboratory
for beginner students.

IMPORTANT RULES:

1. Use ONLY information explicitly supported
by the textbook context.
2. Do not introduce programming concepts,
Python functions, tools, commands, or techniques
that are not supported by the context.
3. Every example must be based on the textbook context.
4. If the context does not provide enough information
for a requested part, keep that part simple and
do not invent textbook-specific information.
5. Keep the explanations beginner-friendly.

The laboratory should contain:
- clear explanations
- important concepts
- simple programming examples
- code examples
- expected outputs
- guided practice
- exercises
- homework
- a short summary

TEXTBOOK CONTEXT:
{context}

TEACHER REQUEST:
{request}
"""
    response = client.responses.parse(
        model="gpt-5-nano",
        input=prompt,
        text_format=ProgrammingLab,
    )

    return response.output_parsed


if __name__ == "__main__":

    request = """
Course: Elementary Programming

Lab Topic: Introduction to Programming

Student Level: Beginner students with no prior
programming experience

Duration: 90 minutes

Requirements:
- Clear explanations
- Simple Python examples
- Expected outputs
- Guided practice
- Exercises
- Homework
- Short summary
"""

    retriever = create_retriever()

    
    documents = retrieve_documents(request)

    lab = generate_lab(
        request,
        documents,
    )
    print("\n==============================")
    print("PROGRAMMING LAB")
    print("==============================")

    print(f"\nLab Title: {lab.lab_title}")
    print(f"Course: {lab.course}")
    print(f"Students: {lab.target_students}")
    print(f"Duration: {lab.duration}")

    print("\nLearning Objectives:")
    for objective in lab.learning_objectives:
        print(f"- {objective}")

    print("\nIntroduction:")
    print(lab.introduction)

    print("\nConcepts:")
    for concept in lab.concepts:
        print(f"- {concept}")

    print("\nExamples:")

    for example in lab.examples:
        print(f"\n### {example.title}")

        print("\nCode:")
        print(example.code)

        print("\nExplanation:")
        print(example.explanation)

        print("\nExpected Output:")
        print(example.expected_output)

    print("\nGuided Practice:")

    for item in lab.guided_practice:
        print(f"- {item}")

    print("\nExercises:")

    for exercise in lab.exercises:
        print(f"\n### {exercise.title}")
        print(f"Difficulty: {exercise.difficulty}")
        print(exercise.instructions)

    print("\nHomework:")

    for homework in lab.homework:
        print(f"- {homework}")

    print("\nSummary:")
    print(lab.summary)