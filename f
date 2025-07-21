import os
import uuid

import ollama
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

load_dotenv()
COLLECTION_NAME = "Default"

# Initialize Ollama client
oclient = ollama.Client(host="localhost")


class RagClient:
    def __init__(self, collection_name=COLLECTION_NAME):
    self.qclint = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )



def inject_text(text, collection_name=COLLECTION_NAME):
    try:
        response = oclient.embeddings(model="nomic-embed-text:latest", prompt=text)
        embeddings = response["embedding"]

        # Create a collection if it doesn't already exist
        if not qclient.collection_exists(collection_name):
            qclient.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=len(embeddings), distance=models.Distance.COSINE
                ),
            )

        # Generate unique ID for the document
        point_id = str(uuid.uuid4())

        qclient.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=embeddings,
                    payload={"text": text},
                )
            ],
        )
        print(f"Text successfully uploaded with ID: {point_id}")

    except Exception as e:
        print(f"Error processing text'{str(e)}")


def inject_documents(filename, collection_name=COLLECTION_NAME):
    """
    Read a file and inject its content into the Qdrant vector database.

    Args:
        filename (str): Path to the file to be processed
    """
    try:
        # Read the file content
        with open(filename, "r", encoding="utf-8") as file:
            text = file.read()

        if not text.strip():
            print(f"Warning: File {filename} is empty")
            return

        # Generate embeddings
        response = oclient.embeddings(model="nomic-embed-text:latest", prompt=text)
        embeddings = response["embedding"]

        # Create a collection if it doesn't already exist
        if not qclient.collection_exists(collection_name):
            qclient.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=len(embeddings), distance=models.Distance.COSINE
                ),
            )

        # Generate unique ID for the document
        point_id = str(uuid.uuid4())

        qclient.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=embeddings,
                    payload={"text": text, "filename": filename},
                )
            ],
        )
        print(f"Document '{filename}' uploaded successfully with ID: {point_id}")

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
    except Exception as e:
        print(f"Error processing file '{filename}': {str(e)}")


if __name__ == "__main__":
    # Example usage - you can replace with any filename
    inject_documents("example.txt")
