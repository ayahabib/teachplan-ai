# TeachPlan-AI 📚🤖

An AI-powered **RAG application** that helps teachers generate structured programming labs from textbook content.

## ✨ How It Works

```text
PDF Upload
    ↓
PyPDFLoader
    ↓
LLM Semantic Chunking
(GPT-5 Nano)
    ↓
OpenAI Embeddings
    ↓
Chroma Vector DB
    ↓
Retriever
    ↓
GPT-5 Nano
    ↓
Programming Lab
```

## 🚀 Features

* 📄 Upload programming textbooks as PDF
* 🧠 LLM-based semantic & structural chunking
* 🔎 Context-aware retrieval with Chroma
* 🤖 AI-generated programming labs
* 🎯 Customize labs by topic, level, duration, and requirements
* 🖥️ Simple Gradio interface

> **Note:** TeachPlan-AI uses LLM-assisted chunking instead of `RecursiveCharacterTextSplitter` to preserve the structure and meaning of textbook content.

## 🛠️ Tech Stack

**Python · LangChain · OpenAI · GPT-5 Nano · Chroma · Gradio**

## ⚙️ Setup

```bash
git clone https://github.com/your-username/teachplan-ai.git
cd teachplan-ai

conda create -n teachplan python=3.12
conda activate teachplan

pip install -r requirements.txt
```

Create `.env`:

```env
OPENAI_API_KEY=your_api_key
```

Run:

```bash
python -m src.app
```

## 👩‍💻 Author

**Aya Habib**
AI Engineer | Machine Learning Engineer | Generative AI Enthusiast
