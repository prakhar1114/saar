"""
Search Video Transcripts using ChromaDB
Local vector search with Google's text-embedding-004 model
"""

import os
import sys
import chromadb
from vertexai.language_models import TextEmbeddingModel
import vertexai
from dotenv import load_dotenv

load_dotenv()


def initialize_vertex_ai():
    """Initialize Vertex AI with project credentials."""
    project_id = os.getenv("GCP_PROJECT_ID")
    if not project_id:
        raise ValueError("Please set GCP_PROJECT_ID in your .env file")

    vertexai.init(project=project_id, location="us-central1")
    return project_id


def search_transcripts(query: str, n_results: int = 5, collection_name: str = "youtube_transcripts"):
    """
    Search through video transcript chunks.

    Args:
        query: The search query (e.g., "stock market", "inflation", etc.)
        n_results: Maximum number of results to return
        collection_name: Name of the ChromaDB collection

    Returns:
        Search results with matching transcript chunks
    """
    # Initialize Vertex AI
    initialize_vertex_ai()

    # Connect to ChromaDB
    client = chromadb.PersistentClient(path="./chroma_db")

    try:
        collection = client.get_collection(collection_name)
    except Exception as e:
        print(f"Error: Collection '{collection_name}' not found.")
        print(f"Please run 'python chromadb_setup.py' first to create the database.")
        return None

    # Initialize embedding model
    model = TextEmbeddingModel.from_pretrained("text-embedding-004")

    # Get query embedding
    query_embedding = model.get_embeddings([query])[0].values

    # Search
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=['documents', 'metadatas', 'distances']
    )

    return results


def print_search_results(results, query: str):
    """Pretty print search results."""
    if not results or not results['documents'] or not results['documents'][0]:
        print("No results found.")
        return

    documents = results['documents'][0]
    metadatas = results['metadatas'][0]
    distances = results['distances'][0]

    print(f"\n{'='*80}")
    print(f"Search Results for: '{query}'")
    print(f"Found {len(documents)} results")
    print(f"{'='*80}\n")

    for i, (doc, metadata, distance) in enumerate(zip(documents, metadatas, distances), 1):
        print(f"Result #{i} (relevance score: {1 - distance:.3f})")
        print(f"{'-'*80}")

        # Video info
        print(f"Video: {metadata.get('video_title', 'N/A')}")
        print(f"Channel: {metadata.get('channel', 'N/A')}")

        # Chunk timing
        start_time = metadata.get('chunk_start_time', 0)
        end_time = metadata.get('chunk_end_time', 0)
        print(f"Time: {start_time}s - {end_time}s ({end_time - start_time}s)")

        # Direct link with timestamp
        video_url = metadata.get('video_url', '')
        if video_url and start_time:
            timestamped_url = f"{video_url}&t={int(start_time)}s"
            print(f"Watch: {timestamped_url}")

        # Published date
        print(f"Published: {metadata.get('video_published_date', 'N/A')}")

        # Content preview (first 300 chars)
        preview = doc[:300] + "..." if len(doc) > 300 else doc
        print(f"\nTranscript Excerpt:\n{preview}")

        print(f"\n")

    print(f"{'='*80}\n")


def main():
    """Main search function."""
    # Get query from command line or use default
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        print("Usage: python chromadb_search.py <your search query>")
        print("\nExample:")
        print("  python chromadb_search.py 'stock market crash'")
        print("  python chromadb_search.py 'inflation rates'")
        return

    print(f"Searching for: '{query}'")

    # Search
    results = search_transcripts(query, n_results=5)

    if results:
        print_search_results(results, query)


if __name__ == "__main__":
    main()
