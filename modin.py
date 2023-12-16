import os
import time
import pdfplumber
import modin.pandas as pd  # Use modin.pandas instead of pandas
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain.embeddings import HuggingFaceEmbeddings

# Set environment variables
os.environ["HUGGINGFACEHUB_API_TOKEN"] = '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'

# Embedding
embeddings = HuggingFaceEmbeddings(model_name="jinaai/jina-embedding-s-en-v1")

# Load data using Modin Pandas
file_path = './rights.csv'
df = pd.read_csv(file_path, encoding='utf-8')

# Assuming the column names are "Question" and "Answer"
question_column_name = 'Question'
answer_column_name = 'Answer'

# Combine questions and answers into a single text column
df['combined_text'] = df[question_column_name] + ' ' + df[answer_column_name]

# Handle NaN values by replacing them with an empty string
df['combined_text'] = df['combined_text'].fillna('')

# Use the combined text directly without the need for the text splitter
docs = df['combined_text']

# Initialize FAISS vector store
db = FAISS.from_texts(docs, embeddings)
start = time.time()

# LLM
repo_id = 'HuggingFaceH4/zephyr-7b-beta'
llm = HuggingFaceHub(
    repo_id=repo_id, model_kwargs={"max_length": 1024, "temperature": 0.5}
)
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(k=2),
    return_source_documents=True,
    verbose=True
)

query = "what does article 14 say"
result = qa(query)
end = time.time()

print(result['result'])
print(end - start)
