from fastapi import FastAPI
import ollama
import chromadb
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)

app = FastAPI()  # Create the FastAPI application

# Connect to the same ChromaDB collection you built in Step 2
client = chromadb.PersistentClient(path="./chroma_db")

ef = OllamaEmbeddingFunction(
    model_name="nomic-embed-text",
    url="http://localhost:11434",
)

collection = client.get_or_create_collection(
    name="personal_profile",
    embedding_function=ef,
)

'''
What does this code do?
This creates a FastAPI application and connects to the same ChromaDB collection you built in Step 2. 
When the server starts, it will have immediate access to your knowledge base.
'''


@app.get("/ask")  # This creates a GET endpoint at /ask
def ask(question: str):  # FastAPI automatically reads "question" from the URL query string
    # Step 1: RETRIEVE - search ChromaDB for the 2 most relevant chunks
    results = collection.query(
        query_texts=[question],  # ChromaDB converts this to a vector and finds similar chunks
        n_results=2,  # Return the top 2 matches
    )
    # Combine the matching chunks into a single string
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

    # Return the answer along with the context so users can verify the source
    return {
        "question": question,
        "answer": response["message"]["content"],
        "context_used": results["documents"][0],
    }

'''
What does this code do?
This single endpoint implements the three steps of RAG that you performed manually in Step 1.
When someone sends a question, it retrieves the 2 most relevant chunks from ChromaDB, augments 
the prompt by combining those chunks with the question, and generates a grounded answer using qwen2.5:0.5b. 
The response includes the context that was used so you can verify the AI's sources.
'''