"""
qa_engine.py – Optimized Core pipeline for Long-text Document QA
Fast version for Streamlit
"""

from pathlib import Path
from typing import List, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    UnstructuredFileLoader,
    TextLoader,
    PyMuPDFLoader,
    Docx2txtLoader,
    CSVLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
)

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch



EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

LLM_MODEL = "google/flan-t5-small"

# Faster chunking
CHUNK_SIZE = 300
CHUNK_OVERLAP = 30

TOP_K = 2

VECTORSTORE_PATH = Path("vectorstore/faiss_index")


LOADER_MAP = {
    ".pdf": PyMuPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
    ".csv": CSVLoader,
    ".pptx": UnstructuredPowerPointLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".xls": UnstructuredExcelLoader,
}


def get_loader(file_path: str):
    ext = Path(file_path).suffix.lower()
    loader_cls = LOADER_MAP.get(ext, UnstructuredFileLoader)
    return loader_cls(file_path)



def load_and_split(file_path: str) -> List:
    loader = get_loader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    chunks = splitter.split_documents(docs)
    return chunks



def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        },
    )



def build_vectorstore(chunks: List, embeddings):
    return FAISS.from_documents(chunks, embeddings)


def save_vectorstore(vectorstore, path: Path = VECTORSTORE_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(path))


def load_vectorstore(embeddings, path: Path = VECTORSTORE_PATH):

    if path.exists():
        return FAISS.load_local(
            str(path),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    return None


def add_to_vectorstore(existing, new_chunks, embeddings):
    new_vs = FAISS.from_documents(new_chunks, embeddings)
    existing.merge_from(new_vs)
    return existing


# LLM

def get_llm():

    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        LLM_MODEL
    )

    text_pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=128,
        do_sample=False,
        truncation=True,
        device=0 if torch.cuda.is_available() else -1,
    )

    return HuggingFacePipeline(pipeline=text_pipe)


# Prompt

QA_PROMPT_TEMPLATE = """
Use only the following context to answer the question.

If the answer is not in the context,
say "I don't know based on the document."

Keep the answer short and accurate.

Context:
{context}

Question:
{question}

Answer:
"""

QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=QA_PROMPT_TEMPLATE,
)


def build_qa_chain(vectorstore, llm):

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_PROMPT},
    )

    return chain


def ask(chain, question: str):

    result = chain.invoke({"query": question})

    answer = result["result"].strip()

    sources = [
        {
            "content": doc.page_content,
            "metadata": doc.metadata,
        }
        for doc in result.get("source_documents", [])
    ]

    return {
        "answer": answer,
        "sources": sources,
    }
