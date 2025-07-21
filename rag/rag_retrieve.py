import os

import ollama
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()
COLLECTION_NAME = "NicheApplications"

# Initialize Ollama client
oclient = ollama.Client(host="localhost")

# Initialize Qdrant client
qclient = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)


def retrieve_documents(query, collection_name=COLLECTION_NAME, limit=3):
    """
    Retrieve similar documents from the Qdrant vector database based on a query.

    Args:
        query (str): Search query text
        limit (int): Number of similar documents to retrieve

    Returns:
        list: List of dictionaries containing document text, filename, and similarity scores
    """
    try:
        # Generate embeddings for the query
        response = oclient.embeddings(model="nomic-embed-text:latest", prompt=query)
        query_embeddings = response["embedding"]

        # Search for similar documents
        search_results = qclient.query_points(
            collection_name=collection_name, query=query_embeddings, limit=limit
        ).points

        # Format results
        results = []
        for result in search_results:
            results.append(
                {
                    "text": result.payload.get("text", "No text available"),
                    "filename": result.payload.get("filename", "Unknown file"),
                    "score": result.score,
                    "id": result.id,
                }
            )

        return results

    except Exception as e:
        print(f"Error retrieving documents: {str(e)}")
        return []


def get_context_for_query(query, collection_name=COLLECTION_NAME, limit=3):
    """
    Get contextual information for a query by retrieving relevant documents.

    Args:
        query (str): Search query text
        limit (int): Number of documents to retrieve for context

    Returns:
        str: Concatenated text from relevant documents
    """
    results = retrieve_documents(query, collection_name, limit)

    if not results:
        return "No relevant context found."

    context_parts = []
    for i, result in enumerate(results, 1):
        context_parts.append(
            f"Document {i} (from {result['filename']}, score: {result['score']:.3f}):\n{result['text']}"
        )

    return "\n\n---\n\n".join(context_parts)


if __name__ == "__main__":
    # Example usage
    query = "Karneeshkar"
    results = retrieve_documents(query)

    print(f"Search results for: '{query}'\n")
    for i, result in enumerate(results, 1):
        print(f"{i}. File: {result['filename']}")
        print(f"   Score: {result['score']:.3f}")
        print(f"   Text preview: {result['text'][:100]}...")
        print()
