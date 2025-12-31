import streamlit as st
import os

from langchain.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

from transformers import pipeline

# ---------------------------------
# CONFIG
# ---------------------------------
SIMILARITY_THRESHOLD = 0.75
DOC_PATH = "data/sample.docx"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "google/flan-t5-base"

st.set_page_config(page_title="Chatbot", layout="centered")
st.title("Chatbot")

# ---------------------------------
# LOAD LLM
# ---------------------------------
@st.cache_resource
def load_llm():
    pipe = pipeline(
        "text2text-generation",
        model=LLM_MODEL,
        max_length=512
    )
    return HuggingFacePipeline(pipeline=pipe)

llm = load_llm()

# ---------------------------------
# LOAD EMBEDDINGS
# ---------------------------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

embeddings = load_embeddings()

# ---------------------------------
# LOAD DOCUMENT & VECTORSTORE
# ---------------------------------
@st.cache_resource
def load_vectorstore():
    if not os.path.exists(DOC_PATH):
        raise FileNotFoundError("data/sample.docx not found")

    loader = Docx2txtLoader(DOC_PATH)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    return FAISS.from_documents(chunks, embeddings)

vectorstore = load_vectorstore()

# ---------------------------------
# USER INPUT
# ---------------------------------
question = st.text_input("Ask something")

if question:
    docs_scores = vectorstore.similarity_search_with_score(
        question, k=3
    )

    use_general_llm = True

    if docs_scores:
        best_score = docs_scores[0][1]
        if best_score < SIMILARITY_THRESHOLD:
            use_general_llm = False

    # ---------------------------------
    # DOCUMENT-BASED (RAG)
    # ---------------------------------
    if not use_general_llm:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever
        )

        result = qa_chain(question)
        st.write(result["result"])

    # ---------------------------------
    # GENERAL KNOWLEDGE FALLBACK
    # ---------------------------------
    else:
        prompt = f"""
Answer the following question clearly and correctly.

Question:
{question}
"""
        response = llm(prompt)
        st.write(response)
