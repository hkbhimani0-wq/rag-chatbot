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
DOC_PATH = "data/sample.docx"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "google/flan-t5-base"
SIMILARITY_THRESHOLD = 0.75

st.set_page_config(page_title="RAG Chatbot", layout="centered")
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
# LOAD VECTORSTORE
# ---------------------------------
@st.cache_resource
def load_vectorstore():
    if not os.path.exists(DOC_PATH):
        raise FileNotFoundError("sample.docx not found in data folder")

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
# SESSION STATE
# ---------------------------------
if "question" not in st.session_state:
    st.session_state.question = ""

if "answer" not in st.session_state:
    st.session_state.answer = ""

# ---------------------------------
# FUNCTION TO PROCESS QUESTION
# ---------------------------------
def process_question():
    question = st.session_state.question
    if not question:
        return

    docs_scores = vectorstore.similarity_search_with_score(question, k=3)
    use_rag = False
    if docs_scores and docs_scores[0][1] < SIMILARITY_THRESHOLD:
        use_rag = True

    if use_rag:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever
        )
        result = qa_chain(question)
        st.session_state.answer = result["result"]
    else:
        prompt = f"""
You are a helpful RAG-based chatbot.
You are built using the Flan-T5 Large Language Model.
If the user asks "Which LLM is used?",
reply exactly:
"This chatbot uses the Flan-T5 LLM."

Answer the question clearly.

Question:
{question}
"""
        st.session_state.answer = llm(prompt)

    # Clear input after processing
    st.session_state.question = ""

# ---------------------------------
# USER INPUT (ENTER KEY ONLY)
# ---------------------------------
st.text_input(
    "Ask something",
    key="question",
    on_change=process_question
)

# ---------------------------------
# DISPLAY ANSWER
# ---------------------------------
if st.session_state.answer:
    st.markdown("### Answer")
    st.write(st.session_state.answer)
