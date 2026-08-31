# 📚 RAG Document Assistant

An AI-powered **Retrieval-Augmented Generation (RAG) Document Assistant** that allows users to upload PDF or TXT documents and ask questions about their content.

The application retrieves relevant document chunks using **FAISS**, reranks them using a **CrossEncoder**, and generates answers using **Ollama Cloud**.

---

## 👨‍💻 Developed By

### **Karamthot Anil Naik**

**B.Tech Computer Science & Engineering**  
**Aspiring Data Scientist**

---

## 🚀 Features

- 📄 Upload PDF and TXT documents
- ✂️ Automatic document text splitting
- 🔢 Ollama-based embeddings
- 🗂️ FAISS vector database
- 🔍 Semantic similarity search
- 📊 CrossEncoder-based reranking
- 🤖 Ollama Cloud LLM
- 📚 Displays retrieved document sources
- 🖥️ Interactive Streamlit interface
- 🔐 Secure API key management using `.env`

---

## 🧠 How RAG Works

This project follows a complete Retrieval-Augmented Generation pipeline:

```text
                📄 Upload Document
                       │
                       ▼
               📚 Document Loader
                       │
                       ▼
                ✂️ Text Splitting
                       │
                       ▼
              🔢 Ollama Embeddings
                       │
                       ▼
                🗂️ FAISS Index
                       │
                       ▼
                  ❓ User Query
                       │
                       ▼
              🔍 Similarity Search
                       │
                       ▼
              📊 CrossEncoder
                Reranking
                       │
                       ▼
             📑 Top Relevant Chunks
                       │
                       ▼
              🤖 Ollama Cloud LLM
                       │
                       ▼
                  💬 Answer
