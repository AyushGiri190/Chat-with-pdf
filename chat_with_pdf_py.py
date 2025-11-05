import os
import sys

# Import required libraries
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import Replicate
from langchain.vectorstores import Chroma # Using the specific import from the notebook

# --- Configuration Constants ---
# NOTE: Replace "Your replicate api" with your actual token before running.
REPLICATE_API_TOKEN = "Your replicate api"
PDF_FILE_PATH = './heart.pdf'
LLM_MODEL = "meta/meta-llama-3-8b"
EMBEDDINGS_MODEL = 'sentence-transformers/all-MiniLM-L6-v2' # Default for HuggingFaceEmbeddings if no model is specified
CHROMA_PERSIST_DIR = './chroma_db'


def get_pdf_text(pdf_path: str) -> str:
    """
    Reads the specified PDF file and extracts all text content page by page.
    (Functionality from original `get_pdf_text` block)
    """
    print(f"Extracting text from: {pdf_file_path}")
    text = ""
    try:
        pdf_reader = PdfReader(pdf_path)
        for page in pdf_reader.pages:
            # Ensure text extraction is handled, even if a page is empty
            text += page.extract_text() or ""
    except FileNotFoundError:
        print(f"Error: The file '{pdf_path}' was not found. Please ensure it exists.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred during PDF reading: {e}")
        sys.exit(1)
        
    return text

def split_text_into_documents(raw_text: str):
    """
    Splits the raw text into chunks using CharacterTextSplitter and converts 
    them into LangChain Document objects.
    (Functionality from original splitting/chunking block)
    """
    print("Splitting text into documents (chunks)...")
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    # The original code splits text, then passes the resulting chunks to create_documents
    chunks = text_splitter.split_text(raw_text)
    texts = text_splitter.create_documents(chunks)
    return texts

def create_and_persist_vectorstore(documents, persist_dir: str) -> Chroma:
    """
    Initializes HuggingFace embeddings, creates a Chroma vector store from the 
    documents, and persists it to disk.
    (Functionality from original embedding/vectorstore block)
    """
    print("Initializing HuggingFace Embeddings...")
    # The notebook implicitly uses the default model for HuggingFaceEmbeddings()
    embeddings = HuggingFaceEmbeddings()
    
    print(f"Creating/Persisting Chroma VectorStore to: {persist_dir}")
    vectorstore = Chroma.from_documents(
        documents, 
        embeddings, 
        collection_name="heart", 
        persist_directory=persist_dir
    )
    vectorstore.persist()
    print("VectorStore successfully persisted.")
    return vectorstore

def setup_qa_chain(vectorstore: Chroma) -> ConversationalRetrievalChain:
    """
    Initializes the Replicate LLM and sets up the ConversationalRetrievalChain.
    (Functionality from original LLM and QA Chain setup blocks)
    """
    print(f"Setting up Conversational Retrieval Chain with LLM: {LLM_MODEL}")
    
    # System prompt and parameters directly from the notebook
    system_prompt = "You are an helpful AI.You do not generate further questions and helpful answer of them"
    
    llm = Replicate(
        model=LLM_MODEL,
        input={
            "temperature": 0.6,
            # Note: Replicate handles prompt/system_prompt slightly differently;
            # passing both as done in the notebook for faithful reproduction
            "prompt": system_prompt, 
            "max_tokens": 2000,
            "system_prompt": system_prompt,
        }
    )

    # Initialize the QA Chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        # Making the vector store retrievable with k=2, as specified
        vectorstore.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True # returns the source document also
    )
    return qa_chain

def start_chat_loop(qa_chain: ConversationalRetrievalChain):
    """
    Runs the interactive chat loop with conversation history management.
    (Functionality from original while True loop)
    """
    chat_history = []
    print("\n" + "="*50)
    print(f"--- Starting PDF Chat with {LLM_MODEL.split('/')[-1]} ---")
    print("Type 'exit', 'quit', or 'q' to end the session.")
    print("="*50)

    while True:
        try:
            query = input('Prompt: ')
            if query.lower() in ["exit", "quit", "q"]:
                print('Exiting')
                # Use sys.exit() as in the original code, but break is often safer in complex scripts
                sys.exit() 

            # Use the retrieval chain to answer the prompt
            result = qa_chain({'question': query, 'chat_history': chat_history})
            print('\nAnswer: ' + result['answer'] + '\n')
            
            # Appending the chat history
            chat_history.append((query, result['answer']))

        except KeyboardInterrupt:
            print('\nExiting')
            sys.exit()
        except Exception as e:
            # Catching potential API errors or other runtime issues
            print(f"An unexpected error occurred in the chat loop: {e}")
            break

def main():
    """
    Main function to orchestrate the PDF chat bot setup and execution.
    """
    # 1. Setup Environment
    if REPLICATE_API_TOKEN == "Your replicate api":
        print("ERROR: Please update the REPLICATE_API_TOKEN constant in the script with your actual token.")
        sys.exit(1)
        
    os.environ['REPLICATE_API_TOKEN'] = REPLICATE_API_TOKEN
    
    # 2. Extract and Process Text
    raw_text = get_pdf_text(PDF_FILE_PATH)
    document_texts = split_text_into_documents(raw_text)

    # 3. Create Vector Store
    vectorstore = create_and_persist_vectorstore(
        documents=document_texts,
        persist_dir=CHROMA_PERSIST_DIR
    )

    # 4. Setup QA Chain
    qa_chain = setup_qa_chain(vectorstore)

    # 5. Start Chat Loop
    start_chat_loop(qa_chain)

if __name__ == '__main__':
    # Added helpful instructions for dependencies
    print("NOTE: Ensure you have installed the required packages: `pip install langchain langchain_community replicate pypdf chromadb sentence-transformers`")
    main()