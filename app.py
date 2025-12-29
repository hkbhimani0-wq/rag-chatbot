import streamlit as st
import os

from langchain.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# ----------------------
# Streamlit Page Setup
# ----------------------
st.set_page_config(page_title="RAG Chatbot", layout="centered")
st.title("RAG-Based Chatbot")

# ----------------------
# DOCX File Path
# ----------------------
DOCX_PATH = "./data/sample.docx"

# ----------------------
# Load DOCX Document
# ----------------------
if not os.path.exists(DOCX_PATH):
    st.error("❌ sample.docx not found! Please upload it in the data folder.")
    st.stop()

loader = Docx2txtLoader(DOCX_PATH)
documents = loader.load()

if not documents:
    st.error("❌ DOCX file is empty.")
    st.stop()

st.success(f"✅ Documents loaded: {len(documents)}")

# ----------------------
# Split Text
# ----------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
docs = text_splitter.split_documents(documents)

# ----------------------
# Embeddings
# ----------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ----------------------
# Vector Store
# ----------------------
vectorstore = FAISS.from_documents(docs, embeddings)

# ----------------------
# Load LLM (FREE – No Token)
# ----------------------
model_name = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=512,
    temperature=0.3
)

llm = HuggingFacePipeline(pipeline=pipe)

# ----------------------
# RAG QA Chain
# ----------------------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# ----------------------
# User Input
# ----------------------
query = st.text_input("Ask a question from the document:")

if query:
    response = qa_chain.run(query)

    if "i don't know" in response.lower() or "not found" in response.lower():
        st.warning("⚠️ This question is outside the document scope.")
    else:
        st.success(response)
