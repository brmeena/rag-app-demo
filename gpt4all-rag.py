import sys
import traceback
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.llms import GPT4All
from langchain.chains import ConversationalRetrievalChain
try:
    persist_directory = "chroma_db"
    model_name = "nomic-embed-text-v1.5.Q4_0.gguf"
    gpt4all_kwargs = {'allow_download': 'False','model_path':'/Users/user/Library/Application Support/nomic.ai/GPT4All'}
    vectorstore = Chroma(persist_directory=persist_directory,embedding_function=GPT4AllEmbeddings(
        model_name=model_name,
        gpt4all_kwargs=gpt4all_kwargs
    ))
    db_retriever=vectorstore.as_retriever()
    docs=db_retriever.invoke("RFID Experience for User")
    #print(docs)
    llm = GPT4All(model="/Users/user/Library/Application Support/nomic.ai/GPT4All/Meta-Llama-3-8B-Instruct.Q4_0.gguf")
    conversation = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=db_retriever,
        return_source_documents=True,
        verbose=False,
    )
    response = conversation({"question": "List all the companies where Elon Musk worked?","chat_history":[]})
    #print(response)
    if 'answer' in response:
        print(response['answer'])
    else:
        print("no answer")
except Exception as e:
    traceback.print_exc(e)
