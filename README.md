# 🤖 AI Document QA System (RAG-based)

An intelligent document question-answering system that allows users to query PDF documents and receive precise, context-based answers using semantic search and NLP.

---

## 🚀 Overview

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline that combines:

* 📄 Document Processing (PDF)
* 🧠 Semantic Search (Embeddings + FAISS)
* 🤖 NLP-based Answer Extraction

Instead of returning raw text, the system provides **clean, relevant answers** based on document content.

---

## ✨ Features

* 📥 Load and process PDF documents
* 🔍 Semantic search using vector embeddings
* ⚡ Fast similarity search with FAISS
* 🧠 NLP-based question answering (HuggingFace)
* 💻 Fully offline (no API required)
* 🔁 Interactive query system

---

## 🛠️ Tech Stack

* Python
* LangChain
* FAISS (Vector Database)
* HuggingFace Transformers
* Sentence Transformers

---

## 📂 Project Structure

```
rag-project/
│── app.py              # Main application
│── data.pdf            # Input document
│── requirements.txt    # Dependencies
│── README.md           # Project documentation
```

---

## ⚙️ Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/ai-document-qa-system.git
cd ai-document-qa-system
```

2. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

---

## ▶️ Usage

Run the application:

```bash
python app.py
```

Then ask questions:

```
Ask: What is AI?
```

---

## 🧠 How It Works

1. 📄 Loads PDF using PyPDFLoader
2. ✂️ Splits text into chunks
3. 🔗 Converts text into embeddings
4. 📊 Stores embeddings in FAISS
5. 🔍 Retrieves relevant chunks
6. 🤖 Uses NLP model to extract final answer

---

## 📌 Example Output

```
Ask: What is AI?

Answer:
Simulation of human intelligence

Source:
Artificial Intelligence (AI) refers to the simulation...
```

---

## 💼 Resume Highlight

**AI-Powered Document QA System**

* Built a semantic document retrieval system using FAISS and embeddings
* Implemented NLP-based question answering using HuggingFace models
* Designed end-to-end RAG pipeline for efficient document querying

---

## 🚀 Future Improvements

* 🌐 Web UI using Streamlit
* 📁 Support for multiple documents
* 💬 Chat-style interface
* ☁️ Deployment (AWS / Vercel)

---

## 👨‍💻 Author

**Shubham Raj**

* GitHub: https://github.com/yourusername
* LinkedIn: https://linkedin.com/in/yourprofile

---

## ⭐ If you like this project

Give it a star ⭐ and share it!
