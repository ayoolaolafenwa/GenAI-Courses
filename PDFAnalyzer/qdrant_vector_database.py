from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from uuid import uuid4
from langchain_openai import OpenAIEmbeddings
from qdrant_client import models
from clients import get_qdrant_client

# load pdfs
def load_pdfs(pdf_dir: Path) -> list:
    docs = []
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        for page in pages:
            page.metadata["document_name"] = pdf_path.name

        docs.extend(pages)

    return docs

# chunk documents
def chunk_documents(docs: list) -> list:
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=150,
    )

    chunks = splitter.split_documents(docs)

    for i, chunk in enumerate(chunks):
        page_num = int(chunk.metadata.get("page", 0)) + 1
        document_name = chunk.metadata.get("document_name", "unknown.pdf")

        chunk.metadata["chunk_id"] = f"chunk_{i}"
        chunk.metadata["page_number"] = page_num
        chunk.metadata["citation"] = f"[{document_name} p.{page_num}]"

    return chunks


# set qdrant vector database path
QDRANT_PATH = Path("qdrant_storage")
COLLECTION_NAME = "document_reports"

EMBEDDING_MODELS = {
    "small": {"name": "text-embedding-3-small", "size": 1536},
    "large": {"name": "text-embedding-3-large", "size": 3072},
}
EMBEDDING_MODEL_KEY = "small"
EMBEDDING_MODEL_NAME = EMBEDDING_MODELS[EMBEDDING_MODEL_KEY]["name"]
EMBEDDING_VECTOR_SIZE = EMBEDDING_MODELS[EMBEDDING_MODEL_KEY]["size"]

# Load embedding model
embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)


# Create collection for storing vector embeddings of chunked documents in the vector database 
def create_collection() -> None:
    client = get_qdrant_client()
    try:
        if client.collection_exists(COLLECTION_NAME):
            return

        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=EMBEDDING_VECTOR_SIZE,
                distance=models.Distance.COSINE,
            ),
        )
    finally:
        client.close()

# Convert Chunked documents to vector embeddings
def create_document_embeddings(chunks):
    texts = [chunk.page_content for chunk in chunks]
    vectors = embedding_model.embed_documents(texts)

    points = []
    for chunk, vector in zip(chunks, vectors):
        payload = {
            "content": chunk.page_content,
            "document_name": chunk.metadata.get("document_name", "unknown.pdf"),
            "page_number": chunk.metadata.get("page_number", 0),
            "chunk_id": chunk.metadata.get("chunk_id", ""),
            "citation": chunk.metadata.get("citation", ""),
            "source": chunk.metadata.get("source", ""),
        }

        points.append(
            models.PointStruct(
                id=str(uuid4()),
                vector=vector,
                payload=payload,
            )
        )

    return points


# upload vector embeddings belonging to chunked documents to qdrant collection
def upload_document_embeddings(pdf_dir: Path) -> dict:
    """
    Step1: Load pdfs
    Step2: chunk pdf documents
    Step3: Embed document chunks
    Step4: Refresh the collection and upload the current chunk set
    """
    # Load pdfs
    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")
    
    pdf_paths = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_paths:
        raise ValueError(f"No PDFs found in {pdf_dir}")

    documents = load_pdfs(pdf_dir)
    if not documents:
        raise ValueError(f"Unable to load PDFs from {pdf_dir}")
    
    # covert pdf documents to chunks
    document_chunks = chunk_documents(documents)
    upload_batch_size = 128
    client = get_qdrant_client()
    try:
        # Refresh the collection before upload so repeated indexing runs stay
        # in sync with the current PDF directory instead of accumulating
        # duplicate or stale vectors.
        if client.collection_exists(COLLECTION_NAME):
            client.delete_collection(COLLECTION_NAME)

        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=EMBEDDING_VECTOR_SIZE,
                distance=models.Distance.COSINE,
            ),
        )

        # create vector embeddings for pdf documents
        document_embeddings = create_document_embeddings(document_chunks)

        # upload vector embeddings to qdrant collection
        for i in range(0, len(document_embeddings), upload_batch_size):
            batch = document_embeddings[i:i + upload_batch_size]
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=batch,
                wait=True,
            )


        return {
            "num_pdfs": len(pdf_paths),
            "num_chunks": len(document_chunks),
            "collection_name": COLLECTION_NAME,
        }
    
    finally:
        client.close()

def get_pdf_names(pdf_dir: Path = Path("docs")) -> list[str]:
    return sorted(path.name for path in pdf_dir.glob("*.pdf"))


# similarity search to handle information retrieval for a query
def similarity_search_per_document(
    query: str,
    per_doc_topk: int = 3,
    max_results: int | None = None,
    score_threshold: float | None = None,
    pdf_dir: Path = Path("docs"),
):
    client = get_qdrant_client()
    try:
        if not client.collection_exists(COLLECTION_NAME):
            return []

        query_vector = embedding_model.embed_query(query)
        document_names = get_pdf_names(pdf_dir)
        merged = []

        for document_name in document_names:
            response = client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                limit= per_doc_topk,
                with_payload=True,
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_name",
                            match=models.MatchValue(value=document_name),
                        )
                    ]
                ),
            )

            for point in response.points:
                point_score = float(point.score or 0.0)
                if score_threshold is not None and point_score < score_threshold:
                    continue

                payload = point.payload or {}
                merged.append(
                    {
                        "document_name": payload.get("document_name", "unknown.pdf"),
                        "page_number": payload.get("page_number", 0),
                        "chunk_id": payload.get("chunk_id", ""),
                        "citation": payload.get("citation", ""),
                        "content": payload.get("content", ""),
                        "score": point_score,
                    }
                )

        merged.sort(key=lambda item: item["score"], reverse=True)
        return merged[:max_results] if max_results else merged

    finally:
        client.close()
