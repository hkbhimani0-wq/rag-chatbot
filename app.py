import streamlit as st
import os
from langchain.document_loaders import TextLoader
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
# Load Documents
# ----------------------
file_path = "./data/sample.txt"
if not os.path.exists(file_path):
    st.error(f"File not found: {file_path}")
    st.stop()

loader = TextLoader(file_path)
documents = loader.load()
st.write("Documents loaded:", len(documents), "document(s)")

# ----------------------
# Split Text
# ----------------------
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# ----------------------
# Embeddings
# ----------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ----------------------
# Vector Store
# ----------------------
vectorstore = FAISS.from_documents(docs, embeddings)

# ----------------------
# Load LLM Locally (No Token Needed)
# ----------------------
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

pipe = pipeline(
    task="text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=512,
    temperature=0.5
)

llm = HuggingFacePipeline(pipeline=pipe)

# ----------------------
# QA Chain
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
    st.success(response)
