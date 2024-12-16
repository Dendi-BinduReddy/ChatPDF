from flask import Flask, request, render_template, jsonify
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline
from dotenv import load_dotenv
import tabulate
import pdfplumber
import re

# Load environment variables
load_dotenv()

# Constants
FAISS_INDEX_PATH = "faiss_index"

# Initialize Flask app
app = Flask(__name__)

# Step 1: Initialize Hugging Face models
if __name__ == '__main__':
    # Use the PyTorch version of the model explicitly
    summarizer = pipeline("summarization", model="sshleifer/distilbart-xsum-12-6", framework="pt")  # Lighter Summarization Model
    qa_model = pipeline("question-answering", model="distilbert-base-uncased")  # Lighter QA Model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 2: Function to summarize extracted data with Hugging Face
def summarize_data_with_model(data):
    """
    Summarize and make the extracted data more readable.
    """
    try:
        summary = summarizer(data, max_length=200, min_length=50, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        return f"An error occurred while summarizing the data: {str(e)}"

# Step 3: Function to extract tables from PDF
def extract_table_from_pdf(pdf_path, page_number):
    """
    Extract table data from the PDF document at a specific page.
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[page_number - 1]  # Page numbers are 0-indexed
            table = page.extract_tables()
            if table:
                return table
            else:
                return None
    except Exception as e:
        return f"Error extracting table: {str(e)}"

# Step 4: Function to format tables into HTML
def format_table_as_html(table_data):
    """
    Format a table (list of lists) into HTML format.
    """
    if not table_data:
        return ""
    html_table = "<table border='1'><thead><tr>"
    headers = table_data[0]
    for header in headers:
        html_table += f"<th>{header}</th>"
    html_table += "</tr></thead><tbody>"
    
    for row in table_data[1:]:
        html_table += "<tr>"
        for cell in row:
            html_table += f"<td>{cell}</td>"
        html_table += "</tr>"
    html_table += "</tbody></table>"
    return html_table

# Step 5: Define the local QA Chain function
def local_qa_chain(query, vectorstore, uploaded_pdf_path):
    # Attempt to find a page number in the query using regex
    page_number = None
    match = re.search(r'page (\d+)', query.lower())  # Search for "page <number>"
    
    if match:
        page_number = int(match.group(1))  # Extract the page number
    else:
        return "No page number found in the query.", "", False, []

    # Now we can safely process the query with the page number
    relevant_docs = vectorstore.similarity_search(query, k=5)
    if not relevant_docs:
        return "No relevant documents found in the database.", "", False, []

    context = " ".join([doc.page_content for doc in relevant_docs])

    # Process and summarize the context
    summarized_answer = summarize_data_with_model(context)
    
    # Handle table data if requested
    table_data = []  # Extract table data if available (for simplicity, assume it's extracted manually)
    is_table_request = "table" in query.lower()

    # If the query asks for table data, format it
    formatted_table = format_table_as_html(table_data) if is_table_request else ""

    return summarized_answer, context, is_table_request, formatted_table

# Step 6: Home Route - Upload PDF
@app.route('/')
def index():
    return render_template('index.html')

# Step 7: Handle File Upload and Processing
# Ensure the uploads directory exists
uploads_dir = os.path.join(os.getcwd(), 'uploads')
if not os.path.exists(uploads_dir):
    os.makedirs(uploads_dir)

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if a file is uploaded
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    
    # Save the uploaded file
    pdf_path = os.path.join(uploads_dir, file.filename)
    file.save(pdf_path)

    # Load and process the PDF
    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load()

    # Split the documents into chunks for better indexing
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30)
    split_documents = text_splitter.split_documents(documents)

    # Create and save FAISS vector store
    vectorstore = FAISS.from_documents(split_documents, embeddings)
    vectorstore.save_local(FAISS_INDEX_PATH)

    return jsonify({"message": "File uploaded and processed successfully."})


# Step 8: Query Route
@app.route('/query', methods=['POST'])
def query():
    if request.method == 'POST':
        query_text = request.form.get('query')
        if not query_text:
            return jsonify({"error": "No query provided"}), 400

        # Load the FAISS vector store
        new_vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

        # Get the uploaded PDF path (assuming only one file is uploaded)
        uploaded_pdf_path = os.path.join(uploads_dir, os.listdir(uploads_dir)[0])

        # Run the QA system
        response, context, is_table_request, table_data = local_qa_chain(query_text, new_vectorstore, uploaded_pdf_path)

        # Prepare the JSON response
        result = {
            "answer": response,
            "context": context,
            "is_table_request": is_table_request,
            "table_data": table_data,
        }
        return jsonify(result)


# Step 9: Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)