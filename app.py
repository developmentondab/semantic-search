import os
import pickle
import faiss
import numpy as np
import io

from PyPDF2 import PdfReader
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
# from langchain.chains import load_qa_with_sources_chain
from langchain_community.llms import OpenAI

# Scopes for the Google Drive API
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
DOCS = []

def authenticate():
    """Authenticate and create a Google Drive API service."""
    creds = None
    token_file = 'token.pickle'

    # Load existing credentials
    if os.path.exists(token_file):
        with open(token_file, 'rb') as token:
            creds = pickle.load(token)

    # Refresh or obtain new credentials if needed
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'C:\python\semantic-search\credentials\credentialsnithin.json', SCOPES)
            creds = flow.run_local_server(port=0)

        # Save credentials for future use
        with open(token_file, 'wb') as token:
            pickle.dump(creds, token)

    # Create the Drive API service
    service = build('drive', 'v3', credentials=creds)
    return service

def download_file(service, file_id, file_name):
    """Download a file from Google Drive."""
    request = service.files().get_media(fileId=file_id)
    fh = io.FileIO(file_name, mode='wb')
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
        print(f"Download {int(status.progress() * 100)}%.")
    print(f"File downloaded: {file_name}")

def get_pdf_data(file_path, num_pages=4):
    reader = PdfReader(file_path)
    full_doc_text = ""
    for page in range(len(reader.pages)):
        current_page = reader.pages[page]
        text = current_page.extract_text()
        full_doc_text += text

    return Document(
        page_content=full_doc_text,
        metadata={"source": file_path}
    )

def list_files_in_folder(service, folder_id):
    """List files in a specific folder in Google Drive and download them."""
    query = f"'{folder_id}' in parents"
    results = service.files().list(
        q=query,
        pageSize=10,
        fields="nextPageToken, files(id, name)"
    ).execute()
    items = results.get('files', [])

    if not items:
        print('No files found in this folder.')
        return []

    print(f'Files in folder (ID: {folder_id}):')
    downloaded_files = []
    for item in items:
        print(f'{item["name"]} ({item["id"]})')
        file_name = item['name']
        download_file(service, item['id'], file_name)
        downloaded_files.append(file_name)

    return downloaded_files

def create_search_index(docs):
    """Create a search index from the documents and save it."""
    global DOCS
    source_chunks = []
    splitter = CharacterTextSplitter(separator=" ", chunk_size=1024, chunk_overlap=0)

    for doc in docs:
        source = get_pdf_data(doc)
        for chunk in splitter.split_text(source.page_content):
            source_chunks.append(Document(page_content=chunk, metadata=source.metadata))

    if not source_chunks:
        print("No Docs")
        return False

    DOCS = source_chunks
    print(f"DOCS length: {len(DOCS)}")

    faiss_index = FAISS.from_documents(source_chunks, OpenAIEmbeddings())
    # with open("search_index.pickle", "wb") as f:
    #     pickle.dump(faiss_index, f)

    # Save the FAISS index using FAISS methods
    save_faiss_index(faiss_index.index, "search_index.faiss")
    
    print("Index saved to search_index.pickle")

def save_faiss_index(index, filename):
    """Save the FAISS index to a file."""
    faiss.write_index(index, filename)
    print(f"FAISS index saved to {filename}")

def load_faiss_index(filename):
    """Load the FAISS index from a file."""
    index = faiss.read_index(filename)
    print(f"FAISS index loaded from {filename}")
    return index

def search_in_faiss_index(faiss_index, query, k=3):
    """Search for the most similar documents in the FAISS index."""
    embeddings = OpenAIEmbeddings()
    query_embedding = embeddings.embed_query(query)
    
    D, I = faiss_index.search(np.array([query_embedding]), k)
    
    return D, I

def retrieve_documents_from_index(indices, docs):
    """Retrieve documents based on indices."""
    print(f"DOCS length: {len(DOCS)}")
    return [DOCS[i] for i in indices]

def answer_question_with_chain(question, faiss_index, docs, k=3):
    """Answer a question using the FAISS index and QA chain."""
    distances, indices = search_in_faiss_index(faiss_index, question, k)
    
    relevant_docs = retrieve_documents_from_index(indices[0], docs)
    
    chain = load_qa_with_sources_chain(OpenAI(temperature=0), verbose=False, chain_type="stuff")
    
    result = chain(
        {
            "input_documents": relevant_docs,
            "question": question,
        },
        return_only_outputs=True,
    )
    
    return result

# Example usage
if __name__ == '__main__':
    folder_id = '1RL9DYuSLvOLPRkv7c_8nMMZ2c4KMeq7X'  # Replace with your folder ID
    
    # Authenticate and create the Drive API service
    service = authenticate()
    
    # List and download files from the specified folder
    downloaded_files = list_files_in_folder(service, folder_id)
   
    if downloaded_files:
        # Create and save the search index from downloaded files
        create_search_index(downloaded_files)

        print("   Files search indes added   ")
        question = "How many docs we have"  # Replace with your actual question
        faiss_index = load_faiss_index("search_index.faiss")
        answer = answer_question_with_chain(question, faiss_index, downloaded_files)
        print(answer)
    else:
        print("No files to index.")
