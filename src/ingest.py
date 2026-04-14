import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def create_vector_store():

    filepath = "../data/college.pdf"

    # Check file exists
    if not os.path.exists(filepath):
        print(f"ERROR: File not found at {filepath}")
        print("Make sure college.pdf is in the project root folder.")
        return

    print("Step 1: Loading file...")

    # Detect if it's actually a real PDF
    with open(filepath, "rb") as f:
        header = f.read(5)

    if header.startswith(b"%PDF-"):
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(filepath)
        print("Detected: Real PDF file")
    else:
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(filepath, encoding="utf-8")
        print("Detected: Not a real PDF, loading as plain text instead")

    documents = loader.load()
    print("Total pages/sections loaded:", len(documents))


    print("\nStep 2: Splitting into chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)
    print("Total chunks:", len(chunks))


    print("\nStep 3: Loading embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    print("Embedding model loaded")


    print("\nStep 4: Creating FAISS vector database...")

    vectorstore = FAISS.from_documents(chunks, embeddings)


    print("\nStep 5: Saving vector database...")

    # Save inside src/ so main.py can find it easily
    save_path = os.path.join(os.path.dirname(__file__), "vectorstore")
    vectorstore.save_local(save_path)

    print(f"\nVector database saved to: {save_path}")
    print("Vector database created successfully!")


if __name__ == "__main__":
    create_vector_store()