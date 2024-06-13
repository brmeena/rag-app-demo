import sys
import traceback
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
try:
    loader = PyMuPDFLoader("data/input.pdf")
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(data)

    persist_directory = "chroma_db"
    #print(all_splits)
    model_name = "nomic-embed-text-v1.5.Q4_0.gguf"
    gpt4all_kwargs = {'allow_download': 'False','model_path':'/Users/user/Library/Application Support/nomic.ai/GPT4All'}
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=GPT4AllEmbeddings(
        model_name=model_name,
        gpt4all_kwargs=gpt4all_kwargs
    ),persist_directory=persist_directory)
    question = "Person with experience in NLPLibrary"
    docs = vectorstore.similarity_search(question)

except Exception as e:
    traceback.print_exc(e)
