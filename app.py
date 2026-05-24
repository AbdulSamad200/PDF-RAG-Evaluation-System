"""
PDF Chatbot with RAGAS Evaluation — Production Ready (2026)
===========================================================

FEATURES
--------
✅ PDF Chatbot using Gemini + FAISS
✅ Persistent FAISS storage
✅ Auto-load saved FAISS indexes after restart
✅ Correct modern RAGAS metrics
✅ Improved retrieval quality (MMR)
✅ Better chunking strategy
✅ Content-based hashing (not filename hashing)
✅ Hallucination detection
✅ Context/source inspection
✅ Cleaner prompts for better RAGAS scores
✅ Stable LangChain + RAGAS integration
"""

import os
import hashlib
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

# =========================================================
# LANGCHAIN
# =========================================================

from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

# =========================================================
# RAGAS
# =========================================================

from ragas import evaluate, EvaluationDataset

from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithoutReference,
)

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# =========================================================
# ENV
# =========================================================

load_dotenv()

# =========================================================
# STREAMLIT CONFIG
# =========================================================

st.set_page_config(
    page_title="PDF Chatbot + RAGAS",
    page_icon="🤖",
    layout="wide",
)

# =========================================================
# CONSTANTS
# =========================================================

EMBEDDING_MODEL = "gemini-embedding-001"

CHAT_MODEL = "gemini-2.5-flash-lite"
EVAL_MODEL = "gemini-2.5-flash-lite"

FAISS_ROOT = "faiss_indexes"

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🔧 Configuration")

env_api_key = os.environ.get("GOOGLE_API_KEY", "")

api_key = st.sidebar.text_input(
    "Google API Key",
    value=env_api_key,
    type="password",
)

if not api_key:
    st.warning("Please enter your Google API Key.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = api_key

# =========================================================
# SESSION STATE
# =========================================================

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "current_index" not in st.session_state:
    st.session_state.current_index = None

# =========================================================
# MODELS
# =========================================================

@st.cache_resource
def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL
    )


@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        temperature=0.0,
    )


# =========================================================
# HASHING
# =========================================================

def compute_file_hash(uploaded_files):
    """
    Hash file CONTENTS, not just names.
    Prevents stale FAISS indexes.
    """

    md5 = hashlib.md5()

    sorted_files = sorted(uploaded_files, key=lambda x: x.name)

    for file in sorted_files:
        md5.update(file.name.encode())
        md5.update(file.getvalue())

    return md5.hexdigest()[:12]


# =========================================================
# PDF EXTRACTION
# =========================================================

def extract_text_from_pdf(file):

    try:
        pdf_reader = PdfReader(file)

        text = "\n".join(
            page.extract_text() or ""
            for page in pdf_reader.pages
        )

        return text

    except Exception as e:
        st.warning(f"Failed to process {file.name}: {e}")
        return ""


# =========================================================
# DOCUMENT BUILDING
# =========================================================

def build_documents(uploaded_files):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    docs = []

    for file in uploaded_files:

        text = extract_text_from_pdf(file)

        if text.strip():

            chunks = splitter.create_documents(
                [text],
                metadatas=[{"source": file.name}],
            )

            docs.extend(chunks)

    return docs


# =========================================================
# FAISS HELPERS
# =========================================================

def get_index_path(file_hash):
    return os.path.join(FAISS_ROOT, file_hash)


def get_saved_indexes():

    indexes = {}

    if not os.path.exists(FAISS_ROOT):
        return indexes

    for folder in os.listdir(FAISS_ROOT):

        folder_path = os.path.join(FAISS_ROOT, folder)

        if not os.path.isdir(folder_path):
            continue

        metadata_file = os.path.join(folder_path, "files.txt")

        if os.path.exists(metadata_file):

            with open(metadata_file, "r", encoding="utf-8") as f:
                files = [line.strip() for line in f.readlines()]

            display_name = ", ".join(files)

            indexes[display_name] = folder_path

    return indexes


# =========================================================
# VECTOR STORE
# =========================================================

def load_or_create_vector_store(uploaded_files):

    embeddings = get_embeddings()

    file_hash = compute_file_hash(uploaded_files)

    index_path = get_index_path(file_hash)

    # -----------------------------------------------------
    # LOAD EXISTING INDEX
    # -----------------------------------------------------

    if os.path.exists(index_path):

        st.sidebar.info("📂 Loading saved FAISS index...")

        # Safe because indexes are locally generated
        vector_store = FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True,
        )

        return vector_store, file_hash

    # -----------------------------------------------------
    # CREATE NEW INDEX
    # -----------------------------------------------------

    docs = build_documents(uploaded_files)

    if not docs:
        return None, None

    st.sidebar.info("⚡ Creating embeddings...")

    vector_store = FAISS.from_documents(
        docs,
        embeddings,
    )

    Path(index_path).mkdir(parents=True, exist_ok=True)

    vector_store.save_local(index_path)

    # Save metadata
    metadata_path = os.path.join(index_path, "files.txt")

    with open(metadata_path, "w", encoding="utf-8") as f:
        for file in uploaded_files:
            f.write(file.name + "\n")

    st.sidebar.success(
        f"✅ FAISS index saved ({len(docs)} chunks)"
    )

    return vector_store, file_hash


# =========================================================
# RETRIEVAL
# =========================================================

def retrieve_context(query, k=5):

    retriever = st.session_state.vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": 20,
            "lambda_mult": 0.7,
        },
    )

    return retriever.invoke(query)


# =========================================================
# PROMPT
# =========================================================

RAG_PROMPT = """
You are a precise RAG assistant.

Use ONLY the provided context.

Rules:
- Do NOT use outside knowledge.
- Keep answers concise and factual.
- Do NOT explain unnecessarily.
- If answer is absent in context, say:
  "I cannot answer this from the provided documents."

Context:
{context}
"""


# =========================================================
# GENERATION
# =========================================================

def generate_answer(question, context_docs):

    llm = get_llm()

    context = "\n\n".join(
        doc.page_content
        for doc in context_docs
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_PROMPT),
        ("human", "{question}"),
    ])

    chain = prompt | llm

    response = chain.invoke({
        "context": context,
        "question": question,
    })

    # Add citations
    sources = sorted(set(
        doc.metadata.get("source", "Unknown")
        for doc in context_docs
    ))

    citation_text = "\n\nSources: " + ", ".join(sources)

    return response.content + citation_text


# =========================================================
# CLEANING
# =========================================================

def clean_text(text):
    return " ".join(text.split())


# =========================================================
# RAGAS EVALUATION
# =========================================================

@st.cache_data(ttl=3600)
def run_ragas_evaluation(
    question,
    answer,
    contexts_tuple,
):

    contexts = list(contexts_tuple)

    evaluator_llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(
            model=EVAL_MODEL,
            temperature=0.0,
        )
    )

    evaluator_embeddings = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL
        )
    )

    metrics = [

        Faithfulness(
            llm=evaluator_llm
        ),

        ResponseRelevancy(
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
        ),

        LLMContextPrecisionWithoutReference(
            llm=evaluator_llm
        ),
    ]

    dataset = EvaluationDataset.from_list([{
        "user_input": question,
        "response": answer,
        "retrieved_contexts": contexts,
    }])

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
    )

    return result.to_pandas()


# =========================================================
# MAIN UI
# =========================================================

st.title("📄 PDF Chatbot with RAGAS Evaluation")

st.markdown(
    """
Upload PDFs, ask questions, and evaluate hallucinations instantly.
"""
)

# =========================================================
# LOAD SAVED INDEXES
# =========================================================

saved_indexes = get_saved_indexes()

if saved_indexes and st.session_state.vector_store is None:

    st.sidebar.subheader("📂 Existing Knowledge Bases")

    selected_index = st.sidebar.selectbox(
        "Load Saved FAISS Index",
        options=list(saved_indexes.keys())
    )

    if st.sidebar.button("Load Selected Index"):

        embeddings = get_embeddings()

        st.session_state.vector_store = FAISS.load_local(
            saved_indexes[selected_index],
            embeddings,
            allow_dangerous_deserialization=True,
        )

        st.session_state.current_index = selected_index

        st.sidebar.success("✅ Saved index loaded successfully")

# =========================================================
# PDF UPLOAD
# =========================================================

st.sidebar.subheader("📚 Upload PDFs")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF files",
    type="pdf",
    accept_multiple_files=True,
)

if uploaded_files:

    with st.sidebar.status(
        "Processing PDFs...",
        expanded=True,
    ) as status:

        vector_store, file_hash = load_or_create_vector_store(
            uploaded_files
        )

        if vector_store:

            st.session_state.vector_store = vector_store
            st.session_state.current_index = file_hash

            status.update(
                label="✅ Ready!",
                state="complete",
                expanded=False,
            )

        else:
            st.sidebar.error(
                "No readable text found in PDFs."
            )

# =========================================================
# CHAT UI
# =========================================================

if st.session_state.vector_store is None:

    st.info(
        "👈 Upload PDFs or load an existing FAISS index."
    )

else:

    st.success("✅ Knowledge base ready")

    query = st.text_input(
        "Ask a question about your documents:",
        placeholder="What is the main conclusion?"
    )

    if query:

        # -------------------------------------------------
        # RETRIEVE
        # -------------------------------------------------

        with st.spinner("Retrieving context..."):

            retrieved_docs = retrieve_context(query)

        # -------------------------------------------------
        # GENERATE
        # -------------------------------------------------

        with st.spinner("Generating answer..."):

            answer = generate_answer(
                query,
                retrieved_docs,
            )

        st.subheader("✅ Generated Answer")

        st.write(answer)

        # -------------------------------------------------
        # CONTEXT VIEWER
        # -------------------------------------------------

        with st.expander("🔍 Retrieved Context"):

            for i, doc in enumerate(retrieved_docs):

                st.markdown(
                    f"### Chunk {i+1}"
                )

                st.caption(
                    f"Source: {doc.metadata.get('source', 'N/A')}"
                )

                st.write(doc.page_content)

                st.divider()

        # -------------------------------------------------
        # RAGAS
        # -------------------------------------------------

        st.subheader("📊 RAGAS Evaluation")

        with st.spinner("Running evaluation..."):

            try:

                clean_contexts = [
                    clean_text(doc.page_content)
                    for doc in retrieved_docs
                ]

                eval_df = run_ragas_evaluation(
                    query,
                    answer,
                    tuple(clean_contexts),
                )

                row = eval_df.iloc[0].to_dict()

                faithfulness = row.get(
                    "faithfulness",
                    0.0,
                )

                relevancy = row.get(
                "answer_relevancy",
                 row.get("response_relevancy", 0.0)
                )

                context_precision = row.get(
                    "llm_context_precision_without_reference",
                    0.0,
                )

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Faithfulness",
                    f"{faithfulness:.2f}",
                )

                col2.metric(
                    "Answer Relevancy",
                    f"{relevancy:.2f}",
                )

                col3.metric(
                    "Context Precision",
                    f"{context_precision:.2f}",
                )

                # -----------------------------------------
                # RAW DATAFRAME
                # -----------------------------------------

                with st.expander("📋 Raw Evaluation Data"):

                    st.dataframe(eval_df)

                # -----------------------------------------
                # HALLUCINATION STATUS
                # -----------------------------------------

                st.subheader("🚨 Hallucination Status")

                if faithfulness < 0.7:

                    st.error(
                        "⚠️ Potential hallucination detected"
                    )

                else:

                    st.success(
                        "✅ Answer is grounded in context"
                    )

            except Exception as e:

                st.error(
                    f"RAGAS evaluation failed: {e}"
                )

                st.exception(e)