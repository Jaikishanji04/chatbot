import logging
import os
import tempfile

from dotenv import load_dotenv

load_dotenv()

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------- config (all overridable via environment variables) ----------
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-4-hydrogen")
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-2-preview")
TOP_K = int(os.getenv("TOP_K", "4"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("doc-qa")

st.set_page_config(page_title="Document QA Chatbot")

if not os.getenv("GOOGLE_API_KEY"):
    st.error("GOOGLE_API_KEY is not set. Configure it in the environment.")
    st.stop()


@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(model=CHAT_MODEL)


llm = get_llm()

# ---------- session state ----------
st.session_state.setdefault("vector_db", None)
st.session_state.setdefault("messages", [])
st.session_state.setdefault("document_uploaded", False)

PROMPT_TEMPLATE = """
You are a medical assistant that answers questions using ONLY the text below.

Rules:
- Base your answer strictly on the provided text.
- If the answer is not in the text, say: "I don't know based on the provided document."
- Do not guess, and do not use outside knowledge to fill gaps.
- Do not give a diagnosis or treatment advice. For medical decisions, tell the user to consult a doctor.

Text:
{context}

Question: {query}
"""


def extract_text(content):
    """Return only the answer text, dropping metadata like signatures."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content)


def document_process(file_bytes: bytes):
    # Per-upload temp file: safe for multiple users, deleted right after use.
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file_bytes)
        path = tmp.name
    try:
        docs = PyPDFLoader(path).load()
    finally:
        os.remove(path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
    st.session_state.vector_db = InMemoryVectorStore.from_documents(chunks, embeddings)
    st.session_state.document_uploaded = True
    logger.info("Processed document: %d pages, %d chunks", len(docs), len(chunks))


st.subheader("Document QA Chatbot")

# ---------- upload ----------
if not st.session_state.document_uploaded:
    file = st.file_uploader(label="Upload a PDF file", type="pdf")
    if file:
        try:
            with st.spinner("Processing document..."):
                document_process(file.getvalue())
        except Exception:
            logger.exception("Document processing failed")
            st.error("Could not process this PDF. Please try another file.")
            st.stop()
        st.rerun()

# ---------- chat ----------
if st.session_state.document_uploaded and st.session_state.vector_db:
    for m in st.session_state.messages:
        st.chat_message(m["role"]).markdown(m["content"])

    query = st.chat_input("Enter your query")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        st.chat_message("user").markdown(query)

        try:
            docs = st.session_state.vector_db.similarity_search(query, k=TOP_K)
            context = "\n\n".join(d.page_content for d in docs)
            with st.spinner("Thinking..."):
                response = llm.invoke(
                    PROMPT_TEMPLATE.format(context=context, query=query)
                )
            answer = extract_text(response.content)
        except Exception:
            logger.exception("LLM call failed")
            answer = "Sorry, something went wrong while answering. Please try again."

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").markdown(answer)
