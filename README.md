# 📚 RAG Based Q&A System

A Retrieval-Augmented Generation (RAG) based Question Answering System that allows users to ask questions from an Industrial Training Report.

## 🎯 Project Objective

The main objective of this project is to develop an AI-based Question Answering System that retrieves relevant information from a document and generates answers using a Large Language Model.

## 🛠️ Technologies Used

- Python
- Streamlit
- Pinecone
- Sentence Transformers
- Gemini
- PyPDF

## 🔄 Working of the System

1. The Industrial Training Report is loaded as a PDF.
2. The PDF text is extracted and divided into smaller chunks.
3. Each text chunk is converted into a numerical vector using the `all-MiniLM-L6-v2` embedding model.
4. The vectors are stored in the Pinecone vector database.
5. The user enters a question in the Streamlit application.
6. The question is converted into a vector.
7. Pinecone searches for the most relevant document chunks.
8. The retrieved information is sent to Gemini.
9. Gemini generates an answer using the retrieved document context.
10. The answer and source information are displayed to the user.

## 🧠 RAG Architecture

PDF Document
↓
Text Extraction
↓
Text Chunking
↓
Embedding Model
↓
Pinecone Vector Database
↓
User Question
↓
Question Embedding
↓
Similarity Search
↓
Relevant Document Chunks
↓
Gemini
↓
Final Answer

## ✨ Features

- Ask questions from the Industrial Training Report
- AI-generated answers
- Document-based responses
- Semantic similarity search
- Source display
- Simple Streamlit interface
- Prevents answers based on outside knowledge when information is unavailable in the document

## 📁 Project Structure

RAG-QA-System/
│
├── app.py
├── upload_to_pinecone.py
├── requirements.txt
├── README.md
├── .env
│
└── documents/
    └── Industrial_training_fixed.pdf

## ▶️ How to Run

Install the required packages:

```bash
python -m pip install -r requirements.txt