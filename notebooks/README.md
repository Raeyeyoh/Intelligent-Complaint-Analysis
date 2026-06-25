Task 1: EDA & Preprocessing — CFPB Complaint Data

Objective: Understand the structure and quality of the complaint data,
filter to the four target products, clean narratives, and save the
processed dataset ready for chunking and embedding in Task 2.

Pipeline

1. Load raw CFPB dataset
2. Initial EDA shape, dtypes, nulls
3. Product distribution analysis
4. Narrative length analysis
5. Filter to 4 target products + drop empty narratives
6. Clean text narratives
7. Save filtered dataset

Task 2: Chunking, Embedding & Vector Store Indexing

Objective: Convert cleaned complaint narratives into a searchable
vector store for the RAG pipeline.

Pipeline

1. Load filtered dataset from Task 1
2. Stratified sample (12,000 complaints)
3. Text chunking — RecursiveCharacterTextSplitter (500 chars, 50 overlap)
4. Embedding — all-MiniLM-L6-v2
5. FAISS index build & persist

Task 3: RAG Core Logic & Evaluation

Objective: Build and evaluate the retrieval-augmented generation
pipeline using the persisted vector store from Task 2.

1. Load vector store + models
2. Test retriever in isolation
3. Test full RAG pipeline
4. Qualitative evaluation on 8 representative questions
   task-4
   What the UI includes
   Feature
   Chat history
   Product filter dropdown
   Example questions  
   Sources panel
   Clear button
   Enter key submit
   Error handling
