# ChromaDB Local Search Setup

## Overview
This project now uses **ChromaDB** for local vector search with Google's **text-embedding-004** model. This approach is much simpler than Vertex AI Search and runs entirely on your machine.

## What You Get
- ✅ **Local vector database** - No complex GCP setup
- ✅ **Google's SOTA embeddings** - Best-in-class semantic search
- ✅ **Fast semantic search** - Find relevant content, not just keywords
- ✅ **Direct video timestamps** - Jump to exact moments in videos

## Setup (One Time)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Fetch YouTube Transcripts
```bash
python fetch_youtube_data.py
```
This creates `video_chunked_transcripts.jsonl` with all your transcript chunks.

### 3. Index into ChromaDB
```bash
python chromadb_setup.py
```
This:
- Loads all transcript chunks
- Generates embeddings using Google's text-embedding-004
- Stores them in a local ChromaDB database (`./chroma_db/`)
- Takes ~2-3 minutes for 1,500 chunks

## Usage

### Search Transcripts
```bash
python chromadb_search.py "your search query"
```

### Examples
```bash
# Search for stock market topics
python chromadb_search.py "stock market crash"

# Search for economic topics
python chromadb_search.py "inflation rates"

# Search for company news
python chromadb_search.py "earnings report"

# Search for specific sectors
python chromadb_search.py "technology sector"
```

## How It Works

1. **Chunking**: Videos are split into 60-second chunks
2. **Embedding**: Each chunk is converted to a vector using text-embedding-004
3. **Storage**: Vectors are stored in ChromaDB with metadata
4. **Search**: Your query is embedded and matched against all chunks
5. **Results**: Top 5 most relevant chunks are returned with:
   - Video title and channel
   - Exact timestamp
   - Direct link to that moment in the video
   - Transcript excerpt
   - Relevance score

## Architecture

```
YouTube Videos
    ↓
[fetch_youtube_data.py]
    ↓
Chunked Transcripts (JSONL)
    ↓
[chromadb_setup.py]
    ↓
Google text-embedding-004 → ChromaDB
    ↓
[chromadb_search.py]
    ↓
Search Results
```

## Files

- `chromadb_setup.py` - One-time setup to index transcripts
- `chromadb_search.py` - Search interface
- `chroma_db/` - Local vector database (excluded from git)
- `video_chunked_transcripts.jsonl` - Source data

## Re-indexing

To add new videos:

1. Run `fetch_youtube_data.py` to get new transcripts
2. Run `chromadb_setup.py` again to re-index everything

The setup script automatically deletes and recreates the collection for a clean slate.

## Cost

- **ChromaDB**: 100% free, runs locally
- **Google Embeddings**:
  - ~1,500 chunks ≈ 150 batches ≈ $0.0001 per 1K chars
  - Total cost: ~$0.10-0.20 for initial indexing
  - Search queries: ~$0.001 per search

Much cheaper than Vertex AI Search!

## Troubleshooting

### "Collection not found"
Run `python chromadb_setup.py` first to create the database.

### "Module not found"
Install dependencies: `pip install -r requirements.txt`

### Slow search
First search is slower due to model initialization. Subsequent searches are faster.

## Next Steps

- Add filters (by channel, date, etc.)
- Implement batch search
- Add a web UI
- Export results to CSV
