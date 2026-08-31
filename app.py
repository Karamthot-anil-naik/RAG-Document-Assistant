
import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder



st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 RAG Document Assistant")
st.caption(
    "Upload a document → Retrieve relevant chunks → Rerank → Ask Ollama"
)


load_dotenv()

OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")

if not OLLAMA_API_KEY:
    st.error("OLLAMA_API_KEY is missing in your .env file.")
    st.stop()



@st.cache_resource
def load_llm():

    return ChatOllama(
        model="gpt-oss:20b-cloud",
        base_url="https://ollama.com",
        client_kwargs={
            "headers": {
                "Authorization": f"Bearer {OLLAMA_API_KEY}"
            }
        }
    )




@st.cache_resource
def load_embeddings():

    return OllamaEmbeddings(
        model="nomic-embed-text"
    )




@st.cache_resource
def load_reranker():

    return CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )




def load_document(uploaded_file):

    
    suffix = os.path.splitext(uploaded_file.name)[1].lower()

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as temp_file:

        temp_file.write(uploaded_file.getvalue())

        temp_path = temp_file.name


    if suffix == ".pdf":

        loader = PyPDFLoader(temp_path)

    
    elif suffix == ".txt":

        loader = TextLoader(
            temp_path,
            encoding="utf-8"
        )

    else:

        os.remove(temp_path)

        raise ValueError(
            "Unsupported file type. Please upload PDF or TXT."
        )


    documents = loader.load()

    # Delete temporary file
    os.remove(temp_path)

    return documents



def create_vectorstore(documents):

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    split_docs = text_splitter.split_documents(documents)

    # Embeddings
    embedding_model = load_embeddings()

    # FAISS
    vectorstore = FAISS.from_documents(
        split_docs,
        embedding_model
    )

    return vectorstore, split_docs




with st.sidebar:

    st.header("⚙️ Settings")

    uploaded_file = st.file_uploader(
        "Upload your document",
        type=["pdf", "txt"]
    )

    st.markdown("---")

    st.write("### RAG Settings")

    retrieval_k = st.slider(
        "Documents to retrieve",
        min_value=5,
        max_value=20,
        value=10
    )

    rerank_k = st.slider(
        "Documents after reranking",
        min_value=1,
        max_value=10,
        value=5
    )

    st.markdown("---")
    st.markdown("### 👨‍💻 Developed By")
    st.write("**Karamthot Anil Naik**")
    st.caption("RAG Document Assistant")

    
if "vectorstore" not in st.session_state:

    st.session_state.vectorstore = None

if "file_name" not in st.session_state:

    st.session_state.file_name = None

if "document_loaded" not in st.session_state:

    st.session_state.document_loaded = False




if uploaded_file is not None:

    # Process only when a new file is uploaded
    if st.session_state.file_name != uploaded_file.name:

        with st.spinner("📄 Processing document..."):

            try:

                # Load document
                documents = load_document(uploaded_file)

                # Create FAISS database
                vectorstore, split_docs = create_vectorstore(
                    documents
                )

                # Save to session
                st.session_state.vectorstore = vectorstore
                st.session_state.file_name = uploaded_file.name
                st.session_state.document_loaded = True

                st.success(
                    f"✅ {uploaded_file.name} loaded successfully!"
                )

                st.info(
                    f"Created {len(split_docs)} document chunks."
                )

            except Exception as e:

                st.error(
                    f"Error processing document: {e}"
                )

    else:

        st.success(
            f"✅ Document ready: {uploaded_file.name}"
        )



if not st.session_state.document_loaded:

    st.info(
        "👈 Upload a PDF or TXT document from the sidebar to begin."
    )

    st.stop()



st.subheader("💬 Ask a question")

query = st.text_input(
    "Question",
    placeholder="What are the key side effects mentioned?"
)


ask_button = st.button(
    "🔎 Ask",
    type="primary"
)




if ask_button:

    if not query.strip():

        st.warning("Please enter a question.")

        st.stop()


    try:


        llm = load_llm()

        reranker = load_reranker()

        vectorstore = st.session_state.vectorstore


      

        with st.spinner("🔍 Retrieving relevant chunks..."):

            retriever = vectorstore.as_retriever(
                search_kwargs={
                    "k": retrieval_k
                }
            )

            retrieved_docs = retriever.invoke(query)


        

        with st.spinner("📊 Reranking documents..."):

            pairs = [
                (query, doc.page_content)
                for doc in retrieved_docs
            ]

            scores = reranker.predict(pairs)

            ranked_docs = sorted(
                zip(retrieved_docs, scores),
                key=lambda x: x[1],
                reverse=True
            )

            top_docs = [
                doc
                for doc, score in ranked_docs[:rerank_k]
            ]


    

        context_parts = []

        for doc in top_docs:

            context_parts.append(
                doc.page_content
            )

        context = "\n\n".join(context_parts)


      
        prompt = f"""
You are a document question-answering assistant.

Answer the question using ONLY the information
provided in the context.

Do not use outside knowledge.

If the answer cannot be found in the context,
say:

"I don't know based on the uploaded document."

Context:
-------------------------
{context}
-------------------------

Question:
{query}

Answer clearly and concisely.
"""


        # ------------------------------------------------------
        # LLM GENERATION
        # ------------------------------------------------------

        with st.spinner("🤖 Generating answer..."):

            response = llm.invoke(prompt)


        # ------------------------------------------------------
        # DISPLAY ANSWER
        # ------------------------------------------------------

        st.subheader("🤖 Answer")

        st.write(response.content)


        # ------------------------------------------------------
        # SHOW SOURCES
        # ------------------------------------------------------

        st.subheader("📚 Retrieved Sources")

        for i, (doc, score) in enumerate(
            ranked_docs[:rerank_k],
            start=1
        ):

            with st.expander(
                f"Source {i} — Reranker Score: {score:.4f}"
            ):

                st.write(doc.page_content)

                if doc.metadata:

                    st.caption(
                        f"Metadata: {doc.metadata}"
                    )


    except Exception as e:

        st.error(
            f"Error while answering the question: {e}"
        )
