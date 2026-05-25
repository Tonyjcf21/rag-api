import os
from fastapi import FastAPI
from pydantic import BaseModel
import ollama
import chromadb
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)

# ── Config ──────────────────────────────────────────────
USE_MOCK_LLM = os.environ.get("USE_MOCK_LLM", "0") == "1"

# ── App + ChromaDB Setup ───────────────────────────────
app = FastAPI()

client = chromadb.PersistentClient(path="./chroma_db")

ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")

ef = OllamaEmbeddingFunction(
    model_name="nomic-embed-text",
    url=ollama_host,
)

collection = client.get_or_create_collection(
    name="personal_profile",
    embedding_function=ef,
)


# ── Models ──────────────────────────────────────────────
class DocumentSubmission(BaseModel):
    user_name: str
    content: str


# ── POST /documents ────────────────────────────────────
@app.post("/documents")
def add_document(submission: DocumentSubmission):
    chunks = [chunk.strip() for chunk in submission.content.split("\n\n") if chunk.strip()]

    collection.add(
        ids=[f"{submission.user_name}-chunk{i}" for i in range(len(chunks))],
        documents=chunks,
        metadatas=[
            {"source": "profile", "user_name": submission.user_name, "chunk_index": i}
            for i in range(len(chunks))
        ],
    )

    return {
        "message": f"Added {len(chunks)} chunks for user '{submission.user_name}'.",
        "user_name": submission.user_name,
        "chunks_added": len(chunks),
    }


# ── GET /ask (with mock mode for CI) ──────────────────
# curl -X GET "http://127.0.0.1:8000/ask" -G --data-urlencode "question=Who likes climbing?"
@app.get("/ask")
def ask(question: str, user: str = None):
    query_params = {
        "query_texts": [question],
        "n_results": 2,
    }

    if user:
        query_params["where"] = {"user_name": user}

    # Step 1: RETRIEVE
    results = collection.query(**query_params)
    context = "\n\n".join(results["documents"][0])

    # If mock mode is on, skip the LLM and return raw context
    if USE_MOCK_LLM:
        return {
            "question": question,
            "answer": context,
            "context_used": results["documents"][0],
        }

    # Step 2: AUGMENT
    augmented_prompt = f"""Use the following context to answer the question.
If the context doesn't contain relevant information, say so.

Context:
{context}

Question: {question}"""

    # Step 3: GENERATE
    response = ollama.chat(
        model="qwen2.5:0.5b",
        messages=[{"role": "user", "content": augmented_prompt}],
    )

    return {
        "question": question,
        "answer": response["message"]["content"],
        "context_used": results["documents"][0],
    }
