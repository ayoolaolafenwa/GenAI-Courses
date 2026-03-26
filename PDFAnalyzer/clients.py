import os
from functools import lru_cache
from openai import OpenAI
from qdrant_client import QdrantClient
from pathlib import Path
from dotenv import load_dotenv

load_dotenv("var.env")

@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_qdrant_client() -> QdrantClient:
    return QdrantClient(path=str(Path("qdrant_storage")))
