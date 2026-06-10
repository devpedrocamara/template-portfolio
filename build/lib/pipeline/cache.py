import chromadb
import hashlib
from typing import Optional
from src.pipeline.rag import get_embedding_function, PERSIST_DIR

CACHE_COLLECTION_NAME = "semantic_cache"
SIMILARITY_THRESHOLD = 0.15 

def get_cache_collection():
    chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
    return chroma_client.get_or_create_collection(
        name=CACHE_COLLECTION_NAME,
        embedding_function=get_embedding_function()
    )

def check_cache(query: str) -> Optional[str]:
    collection = get_cache_collection()
    
    if collection.count() == 0:
        return None

    results = collection.query(
        query_texts=[query],
        n_results=1
    )

    if results['distances'] and results['distances'][0]:
        distance = results['distances'][0][0]
        if distance < SIMILARITY_THRESHOLD:
            return results['metadatas'][0][0]['response']
            
    return None

def save_to_cache(query: str, response: str):
    collection = get_cache_collection()
    doc_id = hashlib.md5(query.encode()).hexdigest()
    
    collection.add(
        ids=[doc_id],
        documents=[query],
        metadatas=[{"response": response}]
    )