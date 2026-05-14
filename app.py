"""
Fast Streamlit UI for Document QA
"""

import os
import sys
import shutil
import tempfile
from pathlib import Path

import streamlit as st

# Add src folder to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from qa_engine import (
    load_and_split,
    get_embeddings,
    build_vectorstore,
    save_vectorstore,
    load_vectorstore,
    add_to_vectorstore,
    get_llm,
    build_qa_chain,
    ask,
    VECTORSTORE_PATH,
)


st.set_page_config(
    page_title="Fast DocQA",
    page_icon="📄",
    layout="wide",
)

st.title("Long-Text Document QA System")



@st.cache_resource
def cached_embeddings():
    return get_embeddings()


@st.cache_resource
def cached_llm():
    return get_llm()



if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "models_loaded" not in st.session_state:
    st.session_state.models_loaded = False



with st.sidebar:

    st.header("⚙️ Setup")

    # Load Models
    if not st.session_state.models_loaded:

        if st.button("Load Models"):

            with st.spinner("Loading embeddings..."):
                st.session_state.embeddings = cached_embeddings()

            with st.spinner("Loading LLM..."):
                st.session_state.llm = cached_llm()

            vs = load_vectorstore(st.session_state.embeddings)

            if vs:
                st.session_state.vectorstore = vs

                st.session_state.qa_chain = build_qa_chain(
                    st.session_state.vectorstore,
                    st.session_state.llm,
                )

            st.session_state.models_loaded = True

            st.success("Models Loaded!")

    else:
        st.success("Models already loaded")

    st.divider()

    # Upload Files
    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "txt", "docx", "md"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.session_state.models_loaded:

        if st.button("Index Documents"):

            all_chunks = []

            progress = st.progress(0)

            for i, uf in enumerate(uploaded_files):

                progress.progress((i + 1) / len(uploaded_files))

                suffix = Path(uf.name).suffix

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=suffix
                ) as tmp:

                    tmp.write(uf.read())
                    tmp_path = tmp.name

                try:

                    chunks = load_and_split(tmp_path)

                    for c in chunks:
                        c.metadata["source_file"] = uf.name

                    all_chunks.extend(chunks)

                finally:
                    os.unlink(tmp_path)

            if all_chunks:

                if st.session_state.vectorstore is None:

                    st.session_state.vectorstore = build_vectorstore(
                        all_chunks,
                        st.session_state.embeddings,
                    )

                else:

                    st.session_state.vectorstore = add_to_vectorstore(
                        st.session_state.vectorstore,
                        all_chunks,
                        st.session_state.embeddings,
                    )

                save_vectorstore(st.session_state.vectorstore)

                st.session_state.qa_chain = build_qa_chain(
                    st.session_state.vectorstore,
                    st.session_state.llm,
                )

                st.success("Documents indexed successfully!")

    st.divider()

    if st.button("Clear Database"):

        st.session_state.vectorstore = None
        st.session_state.qa_chain = None
        st.session_state.chat_history = []

        if VECTORSTORE_PATH.exists():
            shutil.rmtree(VECTORSTORE_PATH.parent)

        st.success("Database cleared")

# Chat

for msg in st.session_state.chat_history:

    if msg["role"] == "user":
        st.chat_message("user").write(msg["text"])

    else:
        st.chat_message("assistant").write(msg["text"])


if st.session_state.qa_chain:

    question = st.chat_input("Ask a question")

    if question:

        st.session_state.chat_history.append({
            "role": "user",
            "text": question,
        })

        with st.spinner("Thinking..."):

            result = ask(
                st.session_state.qa_chain,
                question
            )

        st.session_state.chat_history.append({
            "role": "assistant",
            "text": result["answer"],
        })

        st.rerun()

else:
    st.info("Load models and upload documents first.")