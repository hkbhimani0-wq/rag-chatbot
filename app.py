import streamlit as st
import os
import docx2txt
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
# Data folder check
# ----------------------
data_folder = "./data"
if not os.path.exists(data_folder):
    st.error(f"Data folder not found: {data_folder}")
    st.stop()

# Look for DOCX files in data folder
docx_files = [f for f in os.listdir(data_folder) if f.endswith(".docx")]
if not docx_files:
    st.error("No .docx file found in data folder. Please upload sample.docx")
    st.stop()

# Take the first DOCX file (can extend to multiple)
file_path = os.path.join(data_folder, docx_files[0])

# ----------------------
# Load Documents
# ----------------------
try:
    text = docx2txt.process(file_path)
    if not text.strip():
        st.warning("The document is empty.")
        st.stop()
except Exception as e:
    st.error(f"Error reading DOCX file: {e}")
    st.stop()

st.write(f"Document loaded: {docx_files[0]}")

# ----------------------
# Split Text
# ----------------------
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.create_documents([text])

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
    retriever=vectorstore.as_retriever(),
    return_source_documents=True  # Needed to check document scope
)

# ----------------------
# User Input
# ----------------------
query = st.text_input("Ask a question from the document:")

if query:
    result = qa_chain({"query": query})
    answer = result['result']
    sources = result['source_documents']

    # Check if any document chunk was retrieved
    if not sources or not answer.strip():
        st.warning("⚠ This question is outside the document scope.")
    else:
        st.success(answer)
