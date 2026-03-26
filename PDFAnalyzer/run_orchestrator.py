from uuid import uuid4
from orchestrator_agent import orchestrator_agent
from pathlib import Path
from qdrant_vector_database import upload_document_embeddings
from memory import init_memory

def initialize_app(pdf_dir: Path) -> None:
    init_memory()
    info = upload_document_embeddings(pdf_dir)
    print(
        f"{info['num_pdfs']} PDFs processed into {info['num_chunks']} chunks and uploaded to Qdrant collection {info['collection_name']}"
    )

def chat_with_supervisor(session_id: str | None = None) -> None:
    if session_id is None:
        session_id = str(uuid4())

    print("Analyze your pdfs!! \n")
    print("Use 'q', 'exit', or 'exist' to end chat. \n")
    while True:
        user_query = input("User: ").strip()

        if not user_query:
            continue

        if user_query.lower() in {"q", "exit", "exist"}:
            print("Exiting chat loop.")
            break

        #print("User:", user_query)
        result = orchestrator_agent(user_query, session_id=session_id, verbose=True)
        print("Assistant:", result["final_answer"])
        print()

if __name__ == "__main__":
    pdf_dir = Path("docs")
    initialize_app(pdf_dir)
    chat_with_supervisor()
