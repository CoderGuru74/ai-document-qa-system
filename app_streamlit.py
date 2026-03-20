import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import tempfile

st.set_page_config(page_title="AI Document QA", layout="wide")

# ---------------------------
# UI HEADER
# ---------------------------
st.title("🤖 AI Document Chat")
st.markdown("Upload PDFs and chat with them like ChatGPT 💬")

# ---------------------------
# SESSION STATE (CHAT HISTORY)
# ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "db" not in st.session_state:
    st.session_state.db = None

# ---------------------------
# MULTI PDF UPLOAD
# ---------------------------
uploaded_files = st.file_uploader(
    "📄 Upload PDF(s)",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:
    with st.spinner("⚙️ Processing documents..."):
        all_docs = []

        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(uploaded_file.read())
                temp_path = tmp.name

            loader = PyPDFLoader(temp_path)
            documents = loader.load()
            all_docs.extend(documents)

        text_splitter = CharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )
        docs = text_splitter.split_documents(all_docs)

        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        db = FAISS.from_documents(docs, embeddings)

        st.session_state.db = db

    st.success("✅ Documents processed! Start chatting below.")

# ---------------------------
# DISPLAY CHAT HISTORY
# ---------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------------------------
# USER INPUT (CHAT BOX)
# ---------------------------
query = st.chat_input("Ask something about your document...")

if query:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🤖 Thinking..."):
            if st.session_state.db is not None:
                results = st.session_state.db.similarity_search(query, k=3)

                answer = ""
                for r in results:
                    answer += r.page_content + "\n\n"

                # Trim answer
                answer = answer[:800]

                st.markdown(answer)

                # Save response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

                # Source context (expandable)
                with st.expander("📄 Source Context"):
                    for i, r in enumerate(results):
                        st.markdown(f"**Result {i+1}:**")
                        st.write(r.page_content)
                        st.write("---")

            else:
                st.warning("⚠️ Please upload a PDF first.")