import os
import time
from pathlib import Path
from pypdf import PdfReader
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter

CORPUS_DIR = Path("data/corpus")
PERSIST_DIR = "data/chroma"

def get_embedding_function():
    api_key = os.getenv("GEMINI_API_KEY")
    return OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name="gemini-embedding-001",
        api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

def ingest_and_index():
    docs = []
    for pdf_path in CORPUS_DIR.glob("*.pdf"):
        reader = PdfReader(pdf_path)
        for page_idx, page in enumerate(reader.pages[:30]): 
            text = page.extract_text() or ""
            if text.strip():
                docs.append({
                    "text": text,
                    "source": pdf_path.name,
                    "page": page_idx + 1
                })
                
    if not docs:
        print("Nenhum documento encontrado em data/corpus/")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    
    chunks = []
    for doc in docs:
        for i, chunk in enumerate(splitter.split_text(doc["text"])):
            chunks.append({
                "id": f"{doc['source']}-p{doc['page']}-c{i}",
                "text": chunk,
                "source": doc["source"],
                "page": doc["page"],
                "chunk_idx": i
            })

    chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
    try:
        chroma_client.delete_collection("linux_book")
    except Exception:
        pass
        
    collection = chroma_client.get_or_create_collection(
        name="linux_book",
        embedding_function=get_embedding_function(),
    )
    
    BATCH = 20
    
    print(f"Iniciando indexação de {len(chunks)} chunks...")
    
    for start in range(0, len(chunks), BATCH):
        lote = chunks[start : start + BATCH]
        
        sucesso = False
        while not sucesso:
            try:
                collection.add(
                    ids=[c["id"] for c in lote],
                    documents=[c["text"] for c in lote],
                    metadatas=[
                        {"source": c["source"], "page": c["page"], "chunk_idx": c["chunk_idx"]}
                        for c in lote
                    ],
                )
                print(f"Indexados chunks {start} até {start + len(lote)}...")
                sucesso = True
                time.sleep(2)
                
            except Exception as e:
                if "429" in str(e) or "Quota" in str(e):
                    print("⚠️ Limite de 100 requisições/min atingido.")
                    print("😴 Pausando por 60 segundos para a API resetar a cota...")
                    time.sleep(60)
                else:
                    raise e
                    
    print(f"Sucesso! {collection.count()} chunks indexados com sucesso.")

def retrieve(query: str, k: int = 5) -> list[dict]:
    """Recupera os k trechos mais relevantes do livro técnico"""
    chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
    collection = chroma_client.get_collection(name="linux_book", embedding_function=get_embedding_function())
    
    result = collection.query(query_texts=[query], n_results=k)
    hits = []
    if result["documents"] and result["documents"][0]:
        for i in range(len(result["documents"][0])):
            hits.append({
                "text": result["documents"][0][i],
                "source": result["metadatas"][0][i]["source"],
                "page": result["metadatas"][0][i]["page"],
                "distance": result["distances"][0][i] if result["distances"] else 0.0
            })
    return hits




