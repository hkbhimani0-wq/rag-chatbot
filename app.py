import streamlit as st
import os
from langchain.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

from huggingface_hub import InferenceClient

# ---------------------------------
# CONFIG
# ---------------------------------
SIMILARITY_THRESHOLD = 0.75
DOC_PATH = "data/sample.docx"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# LLM model to use in Hugging Face Inference API (Together provider)
LLM_MODEL = "moonshotai/Kimi-K2-Instruct-0905"

st.set_page_config(page_title="Chatbot", layout="centered")
st.title("Chatbot with Hugging Face Inference API")

# ---------------------------------
# SETUP HUGGINGFACE INFERENCE CLIENT
# ---------------------------------
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    st.error("Please set your HF_TOKEN environment variable in the environment.")
    st.stop()

client = InferenceClient(provider="together", api_key=HF_TOKEN)

# ---------------------------------
# LOAD EMBEDDINGS
# ---------------------------------
@st.cache_resource(show_spinner=True)
def load_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

embeddings = load_embeddings()

# ---------------------------------
# LOAD DOCUMENT & VECTORSTORE
# ---------------------------------
@st.cache_resource(show_spinner=True)
def load_vectorstore():
    if not os.path.exists(DOC_PATH):
        raise FileNotFoundError(f"{DOC_PATH} not found")

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
# FUNCTION TO CALL LLM VIA API
# ---------------------------------
def call_llm_api(prompt: str) -> str:
    try:
        response = client.text_generation(
            model=LLM_MODEL,
            inputs=prompt,
            parameters={"max_new_tokens": 512, "temperature": 0.7},
        )
        return response.generated_text
    except Exception as e:
        return f"Error calling LLM API: {e}"

# ---------------------------------
# USER INPUT & PROCESSING
# ---------------------------------
question = st.text_input("Ask something")

if question:
    # Search vectorstore for similarity scores
    docs_scores = vectorstore.similarity_search_with_score(question, k=3)

    use_general_llm = True

    if docs_scores:
        best_score = docs_scores[0][1]
        # If similarity is above threshold, use document retriever + LLM chain
        if best_score >= SIMILARITY_THRESHOLD:
            use_general_llm = False

    if not use_general_llm:
        # Use vectorstore retriever to get relevant docs
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        retrieved_docs = retriever.get_relevant_documents(question)

        # Combine retrieved docs text as context
        context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
        prompt = f"Use the following context to answer the question clearly:\n\n{context_text}\n\nQuestion: {question}\nAnswer:"
        
        answer = call_llm_api(prompt)
        st.markdown("### Answer (using documents)")
        st.write(answer)

    else:
        # Use general LLM without context
        prompt = f"Answer the following question clearly and correctly:\n\nQuestion: {question}\nAnswer:"
        answer = call_llm_api(prompt)
        st.markdown("### Answer (general knowledge)")
        st.write(answer)
