import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import tempfile

st.set_page_config(page_title="AI Document QA", layout="wide")

st.title("📄 AI Document QA System")
st.write("Upload a PDF and ask questions!")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file is not None:
    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    # Load PDF
    loader = PyPDFLoader(temp_path)
    documents = loader.load()

    # Split text
    text_splitter = CharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    docs = text_splitter.split_documents(documents)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # Vector DB
    db = FAISS.from_documents(docs, embeddings)

    st.success("✅ PDF processed! Ask your question below.")

    query = st.text_input("Ask a question:")

    if query:
        results = db.similarity_search(query, k=3)

        st.subheader("🔍 Relevant Answers")

        for i, r in enumerate(results):
            st.markdown(f"### Result {i+1}")
            st.write(r.page_content)
            st.write("---")