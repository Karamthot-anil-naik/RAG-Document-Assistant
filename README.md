# RAG Document Assistant

An AI-powered Retrieval-Augmented Generation (RAG) Document Assistant that allows users to upload PDF or TXT documents and ask questions about their content.

The application retrieves relevant document chunks using FAISS, reranks them with a CrossEncoder, and generates answers using Ollama Cloud.

## Developed By

### Karamthot Anil Naik

**B.Tech Computer Science and Engineering**  
**Aspiring Data Scientist**

## Features

- Upload PDF and TXT documents
- Automatically split documents into smaller text chunks
- Generate embeddings using Ollama
- Store embeddings in a FAISS vector database
- Perform semantic similarity search
- Rerank retrieved chunks using a CrossEncoder
- Generate answers using Ollama Cloud
- Display relevant document sources
- Interactive Streamlit interface
- Secure API key management using environment variables

## Application Screenshots

### Document Upload

Upload a PDF or TXT document directly through the Streamlit application.

![Document Upload](./upload-dashboard.png)

### RAG Question Answering

Ask questions about the uploaded document and receive answers based on the retrieved context.

![RAG Assistant](./rag.png)

## How RAG Works

```text
                Upload Document
                       |
                       v
               Document Loader
                       |
                       v
                Text Splitting
                       |
                       v
                 Embeddings
                       |
                       v
                FAISS Vector DB
                       |
                       v
                 User Question
                       |
                       v
              Similarity Search
                       |
                       v
              CrossEncoder Reranking
                       |
                       v
              Relevant Document Chunks
                       |
                       v
                Ollama Cloud LLM
                       |
                       v
                    Answer
```

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Web application interface |
| LangChain | RAG application framework |
| Ollama | LLM and embedding models |
| FAISS | Vector similarity search |
| Sentence Transformers | CrossEncoder reranking |
| PyPDF | PDF document processing |
| python-dotenv | Environment variable management |
| uv | Python package and environment management |

## Project Structure

```text
rag-document-assistant/
|
├── app2.py
├── README.md
├── pyproject.toml
├── uv.lock
├── requirements.txt
├── .gitignore
|
├── rag.png
└── upload-dashboard.png
```

## Installation

### 1. Clone the Repository

```bash
git clone <https://github.com/karamthot-anil-naik/rag-document-assistant.git>
```

### 2. Navigate to the Project Directory

```bash
cd rag-document-assistant
```

### 3. Create and Sync the Environment

If you are using `uv`:

```bash
uv sync
```

If the project dependencies are not already defined, install them using:

```bash
uv add streamlit
uv add python-dotenv
uv add langchain-ollama
uv add langchain-community
uv add langchain-text-splitters
uv add faiss-cpu
uv add sentence-transformers
uv add pypdf
```

Alternatively, install the dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the root directory of the project:

```env
OLLAMA_API_KEY=your_ollama_api_key
```

Example:

```env
OLLAMA_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxx
```

Never upload your `.env` file to GitHub.

Add the following entries to your `.gitignore` file:

```gitignore
.env
.venv/
__pycache__/
*.pyc
```

## Ollama Configuration

The project uses the following models:

### Language Model

```text
gpt-oss:20b-cloud
```

### Embedding Model

```text
nomic-embed-text
```

### Reranking Model

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

The language model is accessed through Ollama Cloud.

## Running the Application

Start the Streamlit application using:

```bash
uv run streamlit run app2.py
```

Or, if you are not using `uv`:

```bash
streamlit run app2.py
```

The application will normally be available at:

```text
<http://localhost:8501>
```

## Example Questions

After uploading a document, you can ask questions such as:

```text
What are the key side effects mentioned?

What are the main uses of this medicine?

Who should avoid this medication?

What precautions are mentioned?

What dosage information is provided?

What warnings are mentioned in the document?
```

## RAG Pipeline

### 1. Upload Document

The user uploads a PDF or TXT file through the Streamlit interface.

### 2. Document Loading

The application loads the uploaded file using LangChain document loaders.

For PDF files:

```python
PyPDFLoader
```

For TXT files:

```python
TextLoader
```

### 3. Text Splitting

Large documents are divided into smaller chunks using:

```python
RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
```

Text splitting makes document retrieval more efficient.

### 4. Generate Embeddings

Each document chunk is converted into a numerical vector using:

```text
nomic-embed-text
```

These vectors represent the semantic meaning of the document content.

### 5. Store Embeddings in FAISS

The generated embeddings are stored in a FAISS vector database.

FAISS allows the application to efficiently search for document chunks that are semantically similar to a user's question.

### 6. Retrieve Relevant Documents

When a user asks a question, the application retrieves the most relevant document chunks.

```text
User Question
      |
      v
FAISS Similarity Search
      |
      v
Relevant Document Chunks
```

### 7. CrossEncoder Reranking

The retrieved chunks are reranked using:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

Reranking improves the relevance of the context sent to the language model.

```text
Retrieved Chunks
       |
       v
CrossEncoder
       |
       v
Reranked Chunks
       |
       v
Top Relevant Chunks
```

### 8. Context Construction

The highest-ranked document chunks are combined into a context:

```text
Context = Relevant Document Chunks
```

The application instructs the language model to answer using only the retrieved context.

### 9. Answer Generation

The context and user question are sent to:

```text
gpt-oss:20b-cloud
```

The model then generates the final answer.

## Example RAG Prompt

The application uses a prompt similar to:

```text
You are a document question-answering assistant.

Answer the question using ONLY the information
provided in the context.

Do not use outside knowledge.

If the answer cannot be found in the context, say:

"I don't know based on the uploaded document."
```

This helps reduce unsupported or hallucinated answers.

## Project Objective

The objective of this project is to demonstrate a practical implementation of Retrieval-Augmented Generation.

The system combines:

```text
Document Processing
        +
Text Embeddings
        +
Vector Search
        +
Document Reranking
        +
Large Language Model
        =
Intelligent Document Question Answering
```

## Why This Project?

Traditional language model applications generate answers from their pretrained knowledge.

RAG improves this workflow by retrieving relevant information from an external document and providing that information to the language model as context.

This allows the application to answer questions about documents that were not included in the model's original training data.

## Future Improvements

- Add conversational chat history
- Support multiple uploaded documents
- Add source citations
- Display retrieval and reranking scores
- Implement hybrid search
- Add DOCX support
- Add document analytics
- Deploy the application online
- Improve the user interface and experience
- Cache FAISS indexes for faster startup

## Future Architecture

```text
                    User
                     |
                     v
              Streamlit Interface
                     |
                     v
              Document Upload
                     |
          +----------+----------+
          |                     |
          v                     v
      PDF / TXT            Multiple Documents
          |                     |
          +----------+----------+
                     |
                     v
              Document Loader
                     |
                     v
                 Chunking
                     |
                     v
                Embeddings
                     |
                     v
                  FAISS
                     |
                     v
                Retriever
                     |
                     v
              CrossEncoder
                 Reranker
                     |
                     v
                Top Context
                     |
                     v
             Ollama Cloud LLM
                     |
                     v
               Final Answer
```

## Author

**Karamthot Anil Naik**

B.Tech Computer Science and Engineering

**Interests:**

Data Science • Machine Learning • Generative AI • Natural Language Processing • Data Analytics

## Show Your Support

If you find this project useful, consider giving the repository a star.

## License

This project was created for educational and portfolio purposes.
