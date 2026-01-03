import os
import streamlit as st

from huggingface_hub import InferenceClient
from langchain.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# ---------------------------------
# CONFIG
# ---------------------------------
SIMILARITY_THRESHOLD = 0.75
DOC_PATH = "data/sample.docx"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "moonshotai/Kimi-K2-Instruct-0905"

# ---------------------------------
# HF TOKEN
# ---------------------------------
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    st.error("❌ HF_TOKEN not set in Streamlit Secrets")
    st.stop()

# ---------------------------------
# STREAMLIT UI
# ---------------------------------
st.set_page_config(page_title="Chatbot", layout="centered")
st.title("Chatbot with Hugging Face Inference API")

# ---------------------------------
# HUGGING FACE CLIENT (FIXED)
# ---------------------------------
client = InferenceClient(
    model=LLM_MODEL,
    token=HF_TOKEN
)

# ---------------------------------
# EMBEDDINGS
# ---------------------------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

embeddings = load_embeddings()

# ---------------------------------
# VECTOR STORE
# ---------------------------------
@st.cache_resource
def load_vectorstore():
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
# LLM CALL
# ---------------------------------
def call_llm_api(prompt: str) -> str:
    try:
        response = client.text_generation(
            prompt,
            max_new_tokens=512,
            temperature=0.7,
        )
        return response.strip()
    except Exception as e:
        return f"❌ LLM Error: {e}"

# ---------------------------------
# CHAT LOGIC
# ---------------------------------
question = st.text_input("Ask something")

if question:
    docs_scores = vectorstore.similarity_search_with_score(question, k=3)
    use_general_llm = True

    if docs_scores and docs_scores[0][1] <= SIMILARITY_THRESHOLD:
        use_general_llm = False

    if not use_general_llm:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        docs = retriever.get_relevant_documents(question)
        context = "\n\n".join(d.page_content for d in docs)

        prompt = f"""
Use the following context to answer:

{context}

Question: {question}
Answer:
"""
        st.markdown("### Answer (from document)")
        st.write(call_llm_api(prompt))

    else:
        prompt = f"Question: {question}\nAnswer:"
        st.markdown("### Answer (general)")
        st.write(call_llm_api(prompt))
