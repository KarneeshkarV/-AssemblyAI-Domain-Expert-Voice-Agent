import os
import uuid

import ollama
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

load_dotenv()


class RagClient:
    def __init__(
        self,
        collection_name="Default",
        ollama_host="localhost",
        qdrant_url=None,
        # qdrant_api_key=None,
    ):
        self.collection_name = collection_name
        self.oclient = ollama.Client(host=ollama_host)
        self.qclient = QdrantClient(
            url=qdrant_url or os.getenv("QDRANT_URL") or "http://localhost:6333",
            # api_key=qdrant_api_key or os.getenv("QDRANT_API_KEY"),
        )
        self.embedding_model = "nomic-embed-text:latest"

    def _get_embedding(self, text):
        response = self.oclient.embeddings(model=self.embedding_model, prompt=text)
        return response["embedding"]

    def _ensure_collection_exists(self, vector_size, collection_name):
        if not self.qclient.collection_exists(collection_name=collection_name):
            self.qclient.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size, distance=models.Distance.COSINE
                ),
            )

    def inject_text(self, text, collection_name=None, name=None):
        collection_name = collection_name or self.collection_name
        name = name or "text_document"
        try:
            embeddings = self._get_embedding(text)
            self._ensure_collection_exists(len(embeddings), collection_name)

            point_id = str(uuid.uuid4())

            self.qclient.upsert(
                collection_name=collection_name,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector=embeddings,
                        payload={"text": text, "name": name},
                    )
                ],
            )
            print(
                f"Text successfully uploaded with ID: {point_id} to collection '{collection_name}'"
            )

        except Exception as e:
            print(f"Error processing text: {str(e)}")

    def inject_documents(self, filename, collection_name=None):
        collection_name = collection_name or self.collection_name
        try:
            with open(filename, "r", encoding="utf-8") as file:
                text = file.read()

            if not text.strip():
                print(f"Warning: File {filename} is empty")
                return

            embeddings = self._get_embedding(text)
            self._ensure_collection_exists(len(embeddings), collection_name)

            point_id = str(uuid.uuid4())

            self.qclient.upsert(
                collection_name=collection_name,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector=embeddings,
                        payload={"text": text, "filename": filename, "name": filename},
                    )
                ],
            )
            print(
                f"Document '{filename}' uploaded successfully with ID: {point_id} to collection '{collection_name}'"
            )

        except FileNotFoundError:
            print(f"Error: File '{filename}' not found")
        except Exception as e:
            print(f"Error processing file '{filename}': {str(e)}")

    def retrieve_documents(self, query, collection_name=None, limit=3):
        collection_name = collection_name or self.collection_name
        try:
            query_embeddings = self._get_embedding(query)

            search_results = self.qclient.query_points(
                collection_name=collection_name,
                query_vector=query_embeddings,
                limit=limit,
            ).points

            results = []
            for result in search_results:
                results.append(
                    {
                        "text": result.payload.get("text", "No text available"),
                        "filename": result.payload.get("filename", "Unknown file"),
                        "name": result.payload.get("name", "Unknown"),
                        "score": result.score,
                        "id": result.id,
                    }
                )
            return results

        except Exception as e:
            print(
                f"Error retrieving documents from collection '{collection_name}': {str(e)}"
            )
            return []

    def get_context_for_query(self, query, collection_name=None, limit=3):
        collection_name = collection_name or self.collection_name
        results = self.retrieve_documents(query, collection_name, limit)

        if not results:
            return "No relevant context found."

        context_parts = []
        for i, result in enumerate(results, 1):
            source = result.get('filename', result.get('name', 'Unknown'))
            context_parts.append(
                f"Document {i} (from {source}, score: {result['score']:.3f}):\n{result['text']}"
            )

        return "\n\n---\n\n".join(context_parts)

    def clear_collection(self, collection_name=None):
        """Clear all data from a collection"""
        collection_name = collection_name or self.collection_name
        try:
            if self.qclient.collection_exists(collection_name=collection_name):
                self.qclient.delete_collection(collection_name=collection_name)
                print(f"Collection '{collection_name}' cleared successfully")
            else:
                print(f"Collection '{collection_name}' does not exist")
        except Exception as e:
            print(f"Error clearing collection '{collection_name}': {str(e)}")


if __name__ == "__main__":
    # Example usage
    rag_client = RagClient(collection_name="MyTestCollection")

    # Ingest a document
    rag_client.inject_documents("rag/example.txt")

    # Ingest a piece of text
    rag_client.inject_text("This is a test text about Karneeshkar.")

    # Retrieve documents
    query = "Karneeshkar"
    results = rag_client.retrieve_documents(query)

    print(f"\nSearch results for: '{query}'\n")
    if results:
        for i, result in enumerate(results, 1):
            print(f"{i}. File: {result['filename']}")
            print(f"   Score: {result['score']:.3f}")
            print(f"   Text preview: {result['text'][:100]}...")
            print()
    else:
        print("No results found.")

    # Get context for a query
    context = rag_client.get_context_for_query(query)
    print("\nContext for query:\n")
    print(context)
