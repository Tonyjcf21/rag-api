from fastapi import FastAPI
from pydantic import BaseModel  # Pydantic validates incoming request data
import ollama
import chromadb
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)

app = FastAPI()

# Connect to the same ChromaDB collection you built in Step 2
client = chromadb.PersistentClient(path="./chroma_db")

import os

ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")

ef = OllamaEmbeddingFunction(
    model_name="nomic-embed-text",
    url=ollama_host,
)

collection = client.get_or_create_collection(
    name="personal_profile",
    embedding_function=ef,
)

'''
What does this code do?
This is similar to Step 3, but with one new import. 
Pydantic's BaseModel lets FastAPI automatically validate incoming request data.
'''



# Define the expected shape of incoming data for the POST endpoint
class DocumentSubmission(BaseModel):
    user_name: str  # Who this profile belongs to
    content: str  # The profile text to store


@app.post("/documents")  # POST endpoint - accepts data in the request body
def add_document(submission: DocumentSubmission):
    # Split the submitted profile into chunks by paragraph
    chunks = [chunk.strip() for chunk in submission.content.split("\n\n") if chunk.strip()]

    # Store each chunk in ChromaDB with the user's name attached as metadata
    collection.add(
        ids=[f"{submission.user_name}-chunk{i}" for i in range(len(chunks))],
        documents=chunks,
        metadatas=[
            {"source": "profile", "user_name": submission.user_name, "chunk_index": i}
            for i in range(len(chunks))  # user_name metadata lets us filter by user later
        ],
    )

    return {
        "message": f"Added {len(chunks)} chunks for user '{submission.user_name}'.",
        "user_name": submission.user_name,
        "chunks_added": len(chunks),
    }

'''
What does this code do?
The DocumentSubmission class defines the expected shape of incoming data. 
FastAPI uses this to validate request bodies and auto-generate documentation. 
The endpoint chunks the submitted profile the same way build_knowledge_base.py does, 
but attaches the user's name as metadata on each chunk. This user_name metadata is the key to the whole multi-user system.
'''

# WHEN ASKING QUESTIONS THROUGH TERMINAL: curl -X GET "http://127.0.0.1:8000/ask" -G --data-urlencode "question=Who likes climbing?"
@app.get("/ask")
def ask(question: str, user: str = None):  # user is optional, None means search all profiles
    # Build the query parameters
    query_params = {
        "query_texts": [question],
        "n_results": 2,
    }

    # If a user name was provided, only search that user's chunks
    if user:
        query_params["where"] = {"user_name": user}  # ChromaDB metadata filter

    # Step 1: RETRIEVE - search ChromaDB for the most relevant chunks
    results = collection.query(**query_params)  # ** unpacks the dictionary as keyword arguments
    context = "\n\n".join(results["documents"][0])

    # Step 2: AUGMENT - build a prompt that includes the retrieved context
    augmented_prompt = f"""Use the following context to answer the question.
If the context doesn't contain relevant information, say so.

Context:
{context}

Question: {question}"""

    # Step 3: GENERATE - send the augmented prompt to the local LLM
    response = ollama.chat(
        model="qwen2.5:0.5b",
        messages=[{"role": "user", "content": augmented_prompt}],
    )

    # Return the answer along with metadata about the query
    return {
        "question": question,
        "answer": response["message"]["content"],
        "context_used": results["documents"][0],
        "filtered_by_user": user,  # Shows which user was filtered (or None for all)
    }

'''
What does this code do?
The /ask endpoint now accepts an optional user parameter. When provided, 
ChromaDB's where filter only searches through that user's chunks. Without it, all profiles are searched. 
This is a simple but powerful multi-tenancy pattern
'''