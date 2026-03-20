import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

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
# STEP 3: Embeddings
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
while True:
    query = input("Ask: ")

    if query.lower() in ["exit", "quit"]:
        print("👋 Exiting system...")
        break

    # Retrieve relevant chunks
    results = db.similarity_search(query, k=2)

    print("\n🔍 Top Relevant Results:\n")

    for i, r in enumerate(results):
        print(f"Result {i+1}:\n")
        print(r.page_content[:500])
        print("\n----------------------\n")