"""
ChromaDB Local Setup for Video Transcripts
Uses local ChromaDB with Google's text-embedding-004 model
"""

import json
import os
from typing import List, Dict
import chromadb
from chromadb.config import Settings
from vertexai.language_models import TextEmbeddingModel
import vertexai
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()


def initialize_vertex_ai():
    """Initialize Vertex AI with project credentials."""
    project_id = os.getenv("GCP_PROJECT_ID")
    if not project_id:
        raise ValueError("Please set GCP_PROJECT_ID in your .env file")

    vertexai.init(project=project_id, location="us-central1")
    return project_id


def load_chunks(input_file: str = "video_chunked_transcripts.jsonl") -> List[Dict]:
    """Load transcript chunks from JSONL file."""
    chunks = []

    print(f"Loading chunks from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    print(f"Loaded {len(chunks)} chunks")
    return chunks


def setup_chromadb(chunks: List[Dict], collection_name: str = "youtube_transcripts"):
    """
    Set up ChromaDB with transcript chunks using Google embeddings.

    Args:
        chunks: List of transcript chunk dictionaries
        collection_name: Name for the ChromaDB collection

    Returns:
        ChromaDB collection object
    """
    # Initialize ChromaDB (persistent storage in ./chroma_db directory)
    client = chromadb.PersistentClient(path="./chroma_db")

    # Delete collection if it exists (for clean setup)
    try:
        client.delete_collection(collection_name)
        print(f"Deleted existing collection: {collection_name}")
    except:
        pass

    # Create collection
    collection = client.create_collection(
        name=collection_name,
        metadata={"description": "Video transcript chunks with metadata"}
    )
    print(f"Created collection: {collection_name}")

    # Initialize Google's embedding model
    print("Initializing text-embedding-004 model...")
    model = TextEmbeddingModel.from_pretrained("text-embedding-004")

    # Process chunks in batches for efficiency
    # Reduced batch size to stay within token limits (20k tokens max)
    batch_size = 10
    total_chunks = len(chunks)

    print(f"\nProcessing {total_chunks} chunks...")

    for i in tqdm(range(0, total_chunks, batch_size), desc="Adding chunks"):
        batch = chunks[i:i + batch_size]

        # Prepare batch data
        ids = []
        texts = []
        metadatas = []

        for j, chunk in enumerate(batch):
            chunk_id = f"{chunk['video_id']}_{chunk['chunk_start_time']}_{chunk['chunk_end_time']}"
            ids.append(chunk_id)
            texts.append(chunk['text'])

            # Store metadata
            metadata = {
                'channel': chunk.get('channel', ''),
                'video_title': chunk.get('video_title', ''),
                'video_url': chunk.get('video_url', ''),
                'video_id': chunk.get('video_id', ''),
                'published_at': chunk.get('published_at', ''),
                'chunk_start_time': chunk.get('chunk_start_time', 0),
                'chunk_end_time': chunk.get('chunk_end_time', 0),
                'video_published_date': chunk.get('video_published_date', ''),
            }
            metadatas.append(metadata)

        # Get embeddings for batch
        embeddings_response = model.get_embeddings(texts)
        embeddings = [emb.values for emb in embeddings_response]

        # Add to collection
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )

    print(f"\n✓ Successfully added {total_chunks} chunks to ChromaDB")
    print(f"✓ Database stored in: ./chroma_db")

    return collection


def main():
    """Main setup function."""
    print("="*80)
    print("ChromaDB Setup for Video Transcripts")
    print("="*80)

    # Initialize Vertex AI
    project_id = initialize_vertex_ai()
    print(f"✓ Using GCP Project: {project_id}")

    # Load chunks
    chunks = load_chunks("video_chunked_transcripts.jsonl")

    # Setup ChromaDB
    collection = setup_chromadb(chunks)

    print("\n" + "="*80)
    print("✓ Setup Complete!")
    print("="*80)
    print("\nYou can now search your transcripts using:")
    print("  python chromadb_search.py 'your search query'")
    print("="*80)


if __name__ == "__main__":
    main()
