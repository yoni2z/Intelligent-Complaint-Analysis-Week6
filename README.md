# Intelligent-Complaint-Analysis-Week6

# CrediTrust Complaint Analyzer

## Overview

The **CrediTrust Complaint Analyzer** is an AI-powered tool designed to transform unstructured customer complaint data from the Consumer Financial Protection Bureau (CFPB) into actionable insights for CrediTrust Financial. The system enables product managers, compliance teams, and non-technical stakeholders to quickly identify complaint trends across five key product categories: Credit Cards, Personal Loans, Buy Now, Pay Later (BNPL), Savings Accounts, and Money Transfers. Built as part of the 10 Academy Artificial Intelligence Mastery program, this project leverages Retrieval-Augmented Generation (RAG) and a Gradio interface to provide a user-friendly, interactive chat experience.

Due to storage constraints, the pipeline operates on a sampled dataset of 5,000 complaints, ensuring efficient processing while maintaining representativeness through stratified sampling. The system uses `sentence-transformers/all-MiniLM-L6-v2` for embedding and a lightweight `distilgpt2` model for generation, balancing performance with resource limitations.

This repository contains all code, data, and reports for the project, submitted by **Yonas Zelalem** on **July 10, 2025**.

## Project Tasks

The project is divided into four tasks, each contributing to the complaint analysis pipeline:

1. **Task 1: Exploratory Data Analysis (EDA)**  
   - Analyzed the CFPB dataset (9,608,797 records, 462,436 with non-empty narratives) to understand complaint distribution and narrative characteristics.
   - Filtered for five target products and cleaned narratives for downstream processing.
   - Deliverables: `notebooks/task_eda_preprocessing.ipynb`, `reports/task_1_eda_summary.md`, `reports/product_distribution.png`.

2. **Task 2: Text Chunking, Embedding, and Vector Store Indexing**  
   - Sampled 5,000 complaints (stratified by product) to create ~10,000–15,000 text chunks using `RecursiveCharacterTextSplitter` (chunk_size=500, overlap=50).
   - Generated embeddings with `all-MiniLM-L6-v2` and indexed them using FAISS (`IndexFlatL2`).
   - Deliverables: `src/task_2_chunk_embed_index.py`, `vector_store/sampled_faiss_index.idx`, `vector_store/sampled_chunk_metadata.pkl`, `reports/task_2_chunking_embedding.md`.

3. **Task 3: RAG Pipeline and Evaluation**  
   - Built a RAG pipeline with a FAISS-based retriever and `distilgpt2` generator, using a prompt template to ensure grounded answers.
   - Evaluated on five representative questions, identifying strengths (relevant retrieval) and weaknesses (poor answer coherence due to lightweight LLM and sampled data).
   - Deliverables: `src/rag_pipeline.py`, `src/test_rag.py`, `reports/evaluation_table.md`, `reports/task_3_evaluation.md`.

4. **Task 4: Interactive Chat Interface**  
   - Developed a Gradio interface for querying complaints, displaying AI-generated answers and retrieved sources for trust and verification.
   - Deliverables: `src/app.py`, `reports/task_4_ui.md`, `reports/screenshots/main_ui.png`, `reports/screenshots/question_answer.png`.

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yonas-zelalem/CrediTrust-Complaint-Analyzer.git
   cd CrediTrust-Complaint-Analyzer


Install Dependencies:Ensure Python 3.8–3.10 is installed, then install required packages:
pip install -r requirements.txt

Dependencies include: pandas, numpy, faiss-cpu, sentence-transformers, langchain, transformers, torch, gradio.

Prepare Data:

Ensure data/filtered_complaints.csv exists (generated from Task 1).
If regenerating, run notebooks/task_eda_preprocessing.ipynb to create the sampled dataset.


Generate Vector Store:Run the chunking and indexing script:
python src/task_2_chunk_embed_index.py

This creates vector_store/sampled_faiss_index.idx and vector_store/sampled_chunk_metadata.pkl.


Usage

Run the Gradio Interface:Launch the interactive chat interface:
python src/app.py


Access the interface at http://127.0.0.1:7860 in your browser.
Enter questions (e.g., “Why are people unhappy with BNPL?”) and review AI-generated answers with sources.


Test the RAG Pipeline:Run the evaluation script to test the pipeline:
python src/test_rag.py


Outputs results to the console and saves reports/evaluation_table.md.


Review Reports:

EDA: reports/task_1_eda_summary.md, reports/product_distribution.png
Chunking and Embedding: reports/task_2_chunking_embedding.md
RAG Evaluation: reports/task_3_evaluation.md, reports/evaluation_table.md
UI: reports/task_4_ui.md, reports/screenshots/



Deliverables

Task 1:

notebooks/task_eda_preprocessing.ipynb: EDA and preprocessing code.
reports/task_1_eda_summary.md: Summary of dataset insights.
reports/product_distribution.png: Bar plot of complaint distribution.


Task 2:

src/task_2_chunk_embed_index.py: Script for chunking, embedding, and indexing.
vector_store/sampled_faiss_index.idx: FAISS index for sampled data.
vector_store/sampled_chunk_metadata.pkl: Chunk metadata.
reports/task_2_chunking_embedding.md: Report on chunking and embedding strategy.


Task 3:

src/rag_pipeline.py: RAG pipeline logic (Retriever, Generator, RAGPipeline).
src/test_rag.py: Testing and evaluation script.
reports/evaluation_table.md: Evaluation results for five questions.
reports/task_3_evaluation.md: Analysis of RAG pipeline performance.

Task 4:

src/app.py: Gradio interface for querying complaints.
screenshots/

Limitations and Future Improvements

Limited Answer Quality: The distilgpt2 LLM produces incoherent answers due to its small size. Using a larger model (e.g., mistralai/Mixtral-8x7B-Instruct-v0.1) on a cloud platform would improve coherence.
