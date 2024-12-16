# Chat with PDF Using RAG Pipeline

## Description
This project implements a **Retrieval-Augmented Generation (RAG)** pipeline to enable interactive querying of semi-structured data from multiple PDF files. It combines text extraction, vector embeddings, similarity-based retrieval, and LLM-based response generation to provide accurate, context-aware answers. Users can query specific details, retrieve tabular data, and perform comparisons across multiple PDF files seamlessly.

---

## Key Features
- **PDF Ingestion and Processing**: Extracts and segments text and tabular data from PDF files into manageable chunks.
- **Embeddings and Storage**: Converts chunks into vector embeddings using pre-trained models and stores them in a FAISS vector database for fast retrieval.
- **Query Answering**: Leverages similarity search to retrieve relevant chunks and uses a language model to generate detailed responses.
- **Comparison Handling**: Supports comparison queries by extracting and aggregating relevant data from multiple PDFs.
- **Summarization**: Generates concise summaries of tables or extracted data using state-of-the-art summarization models.

---

## Technologies Used
- **Programming Language**: Python  
- **Frameworks and Libraries**: 
  - Flask  
  - LangChain (Document Loader, Text Splitter)  
  - Hugging Face Transformers (`distilbart-xsum`, `distilbert-base-uncased`, `sentence-transformers/all-MiniLM-L6-v2`)  
  - FAISS for vector similarity search  
  - `pdfplumber` for text and table extraction  
  - `dotenv` for environment variable management  

---

## Use Cases
- Extracting and summarizing structured or semi-structured data from PDFs.
- Providing natural language-based insights into documents for research or business.
- Enabling comparison of specific fields across multiple files (e.g., analyzing trends or differences).

---

## How to Run

### 1. Clone the Repository
```bash
git clone <repository-url>
cd <repository-folder>
pip install -r requirements.txt
python app.py
http://localhost:5000
