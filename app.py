import os

# OPTIONAL: Uncomment if you want OpenAI (and have billing)
# os.environ["OPENAI_API_KEY"] = "your_api_key_here"

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# OPTIONAL (only if using OpenAI)
# from langchain_openai import OpenAI
# from langchain.chains import RetrievalQA

# -------------------------
# STEP 1: Load PDF
# -------------------------
loader = PyPDFLoader("data.pdf")
documents = loader.load()

# -------------------------
# STEP 2: Split Text
# -------------------------
text_splitter = CharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)
docs = text_splitter.split_documents(documents)

# -------------------------
# STEP 3: Embeddings (FREE)
# -------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# -------------------------
# STEP 4: Vector Database
# -------------------------
db = FAISS.from_documents(docs, embeddings)

print("✅ AI Document Search System Ready!\n")

# -------------------------
# STEP 5: Query System
# -------------------------

from transformers import pipeline

# Load lightweight local model
qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

print("✅ Smart AI System Ready!\n")

while True:
    query = input("Ask: ")

    if query.lower() in ["exit", "quit"]:
        break

    results = db.similarity_search(query, k=2)

    context = " ".join([r.page_content for r in results])

    # AI Answer
    answer = qa_model(
        question=query,
        context=context
    )

    print("\n🤖 Answer:\n")
    print(answer["answer"])

    print("\n📄 Source Context:\n")
    print(context[:300])
    print("\n----------------------\n")