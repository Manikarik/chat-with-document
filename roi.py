import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
import os
import tempfile

def main():
    st.set_page_config(page_title="Document Chat Assistant", page_icon="📄", layout="wide")

    with st.sidebar:
        st.title("📄 Document Assistant")
        st.markdown(
            """
            **Features:**
            - Upload PDFs.
            - Summarize content.
            - Ask questions about the documents.
            """
        )
        st.info("Ensure your PDFs are text-based for best results!")

    st.title("Chat with Documents")
    st.markdown("Upload your PDFs and interact with them using AI.")

    # File upload
    uploaded_files = st.file_uploader(
        "📤 Upload your PDF files", accept_multiple_files=True, type=["pdf"]
    )

    if uploaded_files:
        st.success(f"{len(uploaded_files)} file(s) uploaded successfully!", icon="✅")

        documents = []
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name

            loader = PyPDFLoader(temp_file_path)
            documents.extend(loader.load())

            os.remove(temp_file_path)

        with st.spinner("Preparing embeddings..."):
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectorstore = FAISS.from_documents(documents, embeddings)
            retriever = vectorstore.as_retriever()

        qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2", device=-1)
        summarization_pipeline = pipeline("summarization", model="t5-small", device=-1)

        def qa_chain(question):
            """Retrieve and answer questions."""
            docs = retriever.get_relevant_documents(question)
            context = " ".join([doc.page_content for doc in docs[:3]])  # Use top 3 docs
            result = qa_pipeline(question=question, context=context)
            return result.get('answer', "No relevant answer found.")

        def summarize_documents(docs):
            """Summarize documents."""
            combined_text = " ".join([doc.page_content for doc in docs])
            chunk_size = 700 
            chunks = [
                combined_text[i:i + chunk_size] for i in range(0, len(combined_text), chunk_size)
            ]
            summaries = summarization_pipeline(
                chunks, max_length=100, min_length=30, truncation=True
            )
            return " ".join([summary['summary_text'] for summary in summaries])

        tab1, tab2 = st.tabs(["❓ Ask Questions", "📜 Summarize Documents"])

        with tab1:
            st.header("❓ Ask Questions")
            user_query = st.text_input("Ask a question:")
            if user_query:
                with st.spinner("Retrieving answer..."):
                    response = qa_chain(user_query)
                    st.subheader("Answer")
                    st.write(response)

        with tab2:
            st.header("📜 Summarize Documents")
            if st.button("Summarize"):
                with st.spinner("Summarizing documents..."):
                    summary = summarize_documents(documents)
                    st.subheader("Summary")
                    st.write(summary)

    else:
        st.warning("Please upload at least one PDF file to begin.", icon="⚠️")

    # Footer
    st.markdown("---")
    st.write("🔍 Powered by Hugging Face Transformers and LangChain.")

if __name__ == "__main__":
    main()
