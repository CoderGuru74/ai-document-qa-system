import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline
import tempfile

# Page config
st.set_page_config(page_title="AI Document QA", layout="wide")

# Title
st.title("🤖 AI Document QA System")
st.markdown("Ask questions from your uploaded PDF using AI")

# Sidebar
st.sidebar.title("📌 About")
st.sidebar.info(
    "This app uses AI + NLP to answer questions from documents.\n\n"
    "Tech: FAISS, LangChain, HuggingFace"
)

# Upload PDF
uploaded_file = st.file_uploader("📄 Upload your PDF", type="pdf")

# Session state for chat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if uploaded_file is not None:

    with st.spinner("Processing PDF..."):
        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())
            temp_path = tmp.name

        # Load PDF
        loader = PyPDFLoader(temp_path)
        documents = loader.load()

        # Split text
        text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)

        # Embeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # Vector DB
        db = FAISS.from_documents(docs, embeddings)

        # QA model
        qa_model = pipeline(
            "question-answering",
            model="distilbert-base-cased-distilled-squad"
        )

    st.success("✅ PDF processed successfully!")

    st.markdown("---")

    # Input
    query = st.text_input("💬 Ask a question:")

    if query:
        results = db.similarity_search(query, k=2)
        context = " ".join([r.page_content for r in results])

        answer = qa_model(question=query, context=context)

        # Save to history
        st.session_state.chat_history.append((query, answer["answer"], context))

    # Display chat history
    for q, a, c in reversed(st.session_state.chat_history):
        st.markdown(f"**🧑 You:** {q}")
        st.markdown(f"**🤖 AI:** {a}")

        with st.expander("📄 View Source Context"):
            st.write(c[:500])

        st.markdown("---")

else:
    st.info("👆 Upload a PDF to get started")