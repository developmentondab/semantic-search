# langchain_script.py

# from langchain.llms import OpenAI
from langchain_community.llms import OpenAI
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.docstore.document import Document
import requests
# from google.colab import drive
# from langchain_google_community import GoogleDriveLoader
from langchain_googledrive.document_loaders import GoogleDriveLoader
import os
from PyPDF2 import PdfReader
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores.faiss import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
import pickle

# Set your OpenAI API key here
OPENAI_API_KEY = 'YOUR_API_KEY'

# Mount Google Drive
# def mount_drive():
#     drive.mount('/content/drive')

# Define file paths
# gdrive_path = '/content/drive/MyDrive/sleep_pdf/'  # Update this path to your folder

# Function to extract text from PDF and create a Document
def get_pdf_data(file_path, num_pages=1):
    reader = PdfReader(gdrive_path + file_path)
    full_doc_text = ""
    for page in range(len(reader.pages)):
        current_page = reader.pages[page]
        text = current_page.extract_text()
        full_doc_text += text

    return Document(
        page_content=full_doc_text,
        metadata={"source": file_path}
    )

# Function to source documents from Google Drive
def source_docs():
    # loader = GoogleDriveLoader(folder_id="1RL9DYuSLvOLPRkv7c_8nMMZ2c4KMeq7X",
    #                       credentials_path="../../desktop_credetnaisl.json")
    loader = GoogleDriveLoader(
        folder_id="1RL9DYuSLvOLPRkv7c_8nMMZ2c4KMeq7X",
        file_types=["document"],
        # load_extended_matadata=True,
        recursive=False,
        num_results=2,  # Maximum number of file to load
        credentials_path="C:\python\semantic-search\credentials\credentialsnithin.json",
        token_path="C:\python\semantic-search\credentials\token.json"
    )
    docs = loader.load()
    print("**************************")
    print(docs)
    return docs
    # return [get_pdf_data(file) for file in os.listdir(gdrive_path)]

# Function to create and save the search index
def search_index(source_docs):
    source_chunks = []
    splitter = CharacterTextSplitter(separator=" ", chunk_size=1024, chunk_overlap=0)

    for source in source_docs:
        for chunk in splitter.split_text(source.page_content):
            source_chunks.append(Document(page_content=chunk, metadata=source.metadata))
    
    if not source_chunks:
        print("No Docs")
        return False
    
    # Save the index to a pickle file
    with open("search_index.pickle", "wb") as f:
        pickle.dump(FAISS.from_documents(source_chunks, OpenAIEmbeddings()), f)

# Load the QA chain
chain = load_qa_with_sources_chain(OpenAI(temperature=0), verbose=False, chain_type="stuff")

# Function to get and print the answer to a question
def print_answer(question):
    if question == "your_question_here":
        return True
    
    with open("search_index.pickle", "rb") as f:
        search_index = pickle.load(f)
    
    result = chain(
        {
            "input_documents": search_index.similarity_search(question, k=3),
            "question": question,
        },
        return_only_outputs=True,
    )
    
    print(result["output_text"])

if __name__ == "__main__":
    # mount_drive()
    sources = source_docs()
    search_index(sources)
    try:
        question = "your_question_here"  # Replace with your actual question
        print_answer(question)
    except Exception as e:
        print(f"Unexpected error: {e}")
