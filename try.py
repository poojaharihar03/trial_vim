import os
import time
import pdfplumber
# from langchain.document_loaders.base import Document
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS, Chroma
from langchain.llms import HuggingFaceHub
# from streamlit_chat import message
from langchain.embeddings import HuggingFaceEmbeddings

# Set environment variables
os.environ["HUGGINGFACEHUB_API_TOKEN"] = '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'

# Embedding
embeddings = HuggingFaceEmbeddings(model_name="jinaai/jina-embedding-s-en-v1")


from langchain.document_loaders.csv_loader import CSVLoader
import os
# file_path = os.path.abspath('./rights.csv')
loader = CSVLoader(file_path='./rights.csv', encoding='utf-8')

documents = loader.load()


# Text splitter
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200
)

docs = text_splitter.split_documents(documents)
db = FAISS.from_documents(docs, embeddings)
start=time.time()
# LLM
repo_id = 'HuggingFaceH4/zephyr-7b-beta'
llm = HuggingFaceHub(
    repo_id=repo_id, model_kwargs={"max_length": 1024, "temperature": 0.5}
)
qa = RetrievalQA.from_chain_type(llm=llm,
                                 chain_type="stuff",
                                 retriever=db.as_retriever(k=2),
                                 return_source_documents=True,
                                 verbose = True)

query = "what does article 14 says"
result = qa(query)
end=time.time()
print(result['result'])

print(end-start)