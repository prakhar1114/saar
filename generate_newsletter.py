"""
AI-Powered Newsletter Generator
Generates newsletter articles from YouTube video transcripts using ChromaDB and Gemini AI
"""

import os
import re
import time
from datetime import datetime
from typing import List, Dict
import chromadb
from vertexai.language_models import TextEmbeddingModel
import vertexai
from dotenv import load_dotenv
from google import genai
from twilio.rest import Client

load_dotenv()


def setup_gemini_client():
    """Initialize Gemini client using GEMINI_API_KEY from .env"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found in .env file. "
            "Please add your Gemini API key to the .env file."
        )

    # Initialize client with API key from environment
    client = genai.Client(api_key=api_key)

    return client


def initialize_vertex_ai():
    """Initialize Vertex AI with project credentials."""
    project_id = os.getenv("GCP_PROJECT_ID")
    if not project_id:
        raise ValueError("Please set GCP_PROJECT_ID in your .env file")

    vertexai.init(project=project_id, location="us-central1")
    return project_id


def aggregate_search_results(keywords: List[str], n_results_per_keyword: int = 10) -> List[Dict]:
    """
    Search ChromaDB for multiple keywords and deduplicate results.

    Args:
        keywords: List of search keywords
        n_results_per_keyword: Number of results to fetch per keyword

    Returns:
        List of unique chunks with metadata, sorted by relevance
    """
    # Initialize Vertex AI
    initialize_vertex_ai()

    # Connect to ChromaDB
    client = chromadb.PersistentClient(path="./chroma_db")

    try:
        collection = client.get_collection("youtube_transcripts")
    except Exception as e:
        raise RuntimeError(
            f"Error: Collection 'youtube_transcripts' not found.\n"
            f"Please run 'python chromadb_setup.py' first to create the database."
        )

    # Initialize embedding model
    model = TextEmbeddingModel.from_pretrained("text-embedding-004")

    # Store all results with their matching keywords
    all_results = {}  # chunk_id -> chunk_data

    print(f"Searching ChromaDB for {len(keywords)} keyword(s)...")

    for keyword in keywords:
        print(f"  - Searching: {keyword}")

        # Generate embedding for this keyword
        query_embedding = model.get_embeddings([keyword])[0].values

        # Query ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results_per_keyword,
            include=['documents', 'metadatas', 'distances']
        )

        # Process results
        if results and results['documents'] and results['documents'][0]:
            documents = results['documents'][0]
            metadatas = results['metadatas'][0]
            distances = results['distances'][0]

            for doc, metadata, distance in zip(documents, metadatas, distances):
                # Create unique chunk_id
                chunk_id = f"{metadata['video_id']}_{metadata['chunk_start_time']}"

                # If this chunk is new or has better relevance, update it
                relevance_score = 1 - distance

                if chunk_id not in all_results or all_results[chunk_id]['relevance_score'] < relevance_score:
                    all_results[chunk_id] = {
                        'chunk_id': chunk_id,
                        'text': doc,
                        'metadata': metadata,
                        'relevance_score': relevance_score,
                        'matching_keywords': [keyword]
                    }
                else:
                    # Just add the keyword to the existing chunk
                    if keyword not in all_results[chunk_id]['matching_keywords']:
                        all_results[chunk_id]['matching_keywords'].append(keyword)

    # Convert to list and sort by relevance
    unique_chunks = list(all_results.values())
    unique_chunks.sort(key=lambda x: x['relevance_score'], reverse=True)

    print(f"\nFound {len(unique_chunks)} unique transcript chunks")

    return unique_chunks


def build_article_prompt(chunks: List[Dict], target_language: str, keywords: List[str]) -> str:
    """
    Build the prompt for Gemini to generate the article.

    Args:
        chunks: List of transcript chunks
        target_language: Language for the article
        keywords: Search keywords used

    Returns:
        Complete prompt string
    """
    # Build source material section
    source_material = []
    for i, chunk in enumerate(chunks, 1):
        metadata = chunk['metadata']
        text = chunk['text']

        # Format published date
        published_date = metadata.get('video_published_date', metadata.get('published_at', 'Unknown'))

        source_entry = f"""[{i}] Video: "{metadata.get('video_title', 'Unknown')}" | Channel: {metadata.get('channel', 'Unknown')} | Date: {published_date}
Transcript: {text}
"""
        source_material.append(source_entry)

    source_text = "\n".join(source_material)

    # Build the complete prompt
    prompt = f"""You are a professional news article writer. Your task is to write a comprehensive article based on video transcript excerpts.

SEARCH KEYWORDS: {', '.join(keywords)}
TARGET LANGUAGE: {target_language}

SOURCE MATERIAL:
Below are {len(chunks)} video transcript excerpts. Each is numbered [1], [2], [3], etc.

{source_text}

INSTRUCTIONS:
1. Write a comprehensive news article synthesizing this information
2. Structure with: headline, introduction, 2-4 sections with subheadings, conclusion
3. CRITICAL: Cite sources using [1], [2], [3] whenever you mention information from the transcripts
4. Write the ENTIRE article in {target_language}
5. Be objective and journalistic in tone
6. If transcripts conflict or present different perspectives, present multiple viewpoints
7. Use the citations frequently - every major point should be cited
8. Make the article engaging and well-structured

OUTPUT FORMAT:
# [Article Headline in {target_language}]

[Introduction paragraph with citations like [1], [2]]

## Section 1: [Subheading in {target_language}]
[Content with citations [3], [4], etc.]

## Section 2: [Subheading in {target_language}]
[Content with citations]

## Conclusion
[Summary paragraph]

Remember: Write EVERYTHING in {target_language} and cite sources using [1], [2], [3] format."""

    return prompt


def generate_article_with_gemini(client, prompt: str) -> str:
    """
    Generate article using Gemini with error handling and retries.

    Args:
        client: Gemini Client instance
        prompt: The article generation prompt

    Returns:
        Generated article text with citations
    """
    max_retries = 3
    retry_delay = 2  # seconds

    for attempt in range(max_retries):
        try:
            print(f"Calling Gemini API (attempt {attempt + 1}/{max_retries})...")

            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=prompt,
            )

            # Check if response was generated
            if not response or not response.text:
                print("Warning: Response may have been blocked or empty")
                if hasattr(response, 'prompt_feedback'):
                    print(f"Prompt feedback: {response.prompt_feedback}")
                if hasattr(response, 'candidates') and response.candidates:
                    print(f"Candidate feedback: {response.candidates[0].finish_reason}")

                # Try to get partial response
                if hasattr(response, 'parts') and response.parts:
                    return str(response.parts)
                else:
                    raise ValueError("No content generated by Gemini")

            print("✓ Article generated successfully")
            return response.text

        except Exception as e:
            error_msg = str(e)

            # Check for rate limit errors
            if "rate limit" in error_msg.lower() or "quota" in error_msg.lower():
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    print(f"Rate limit hit. Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise RuntimeError(f"Rate limit exceeded after {max_retries} attempts")

            # Check for token limit errors
            elif "token" in error_msg.lower() and "limit" in error_msg.lower():
                raise RuntimeError(
                    "Token limit exceeded. Try reducing n_results_per_keyword or "
                    "using fewer/shorter keywords."
                )

            # Other errors
            else:
                if attempt < max_retries - 1:
                    print(f"Error: {error_msg}. Retrying...")
                    time.sleep(retry_delay)
                    continue
                else:
                    raise RuntimeError(f"Failed to generate article after {max_retries} attempts: {error_msg}")

    raise RuntimeError("Failed to generate article")


def replace_citations_with_video_clips(article: str, chunks: List[Dict]) -> str:
    """
    Replace citation numbers [1], [2], [3] with embedded video clips.

    Args:
        article: Article text with [1], [2], [3] citations
        chunks: List of transcript chunks (in same order as citations)

    Returns:
        Article with citations replaced by video embeds
    """
    # Track which citations have been embedded (first occurrence only)
    embedded_citations = set()

    def replace_citation(match):
        citation_num = int(match.group(1))

        # Check if valid citation number
        if citation_num < 1 or citation_num > len(chunks):
            return match.group(0)  # Return unchanged if invalid

        chunk = chunks[citation_num - 1]
        metadata = chunk['metadata']

        video_id = metadata.get('video_id', '')
        video_url = metadata.get('video_url', '')
        video_title = metadata.get('video_title', 'Unknown Video')
        channel = metadata.get('channel', 'Unknown Channel')
        start_time = metadata.get('chunk_start_time', 0)
        end_time = metadata.get('chunk_end_time', 0)
        published_date = metadata.get('video_published_date', metadata.get('published_at', 'Unknown'))

        # Only embed the first occurrence of each citation
        if citation_num not in embedded_citations:
            embedded_citations.add(citation_num)

            # Create YouTube thumbnail URL
            thumbnail_url = f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"

            # Create video embed HTML with lazy-loaded iframe
            embed_html = f"""
<div class="video-clip" data-citation="{citation_num}" data-video-id="{video_id}" data-start-time="{int(start_time)}">
    <div class="video-container">
        <div class="video-thumbnail" style="background-image: url('{thumbnail_url}');">
            <div class="play-button" onclick="loadVideo(this)">
                <svg viewBox="0 0 68 48" width="68" height="48">
                    <path d="M66.52,7.74c-0.78-2.93-2.49-5.41-5.42-6.19C55.79,.13,34,0,34,0S12.21,.13,6.9,1.55 C3.97,2.33,2.27,4.81,1.48,7.74C0.06,13.05,0,24,0,24s0.06,10.95,1.48,16.26c0.78,2.93,2.49,5.41,5.42,6.19 C12.21,47.87,34,48,34,48s21.79-0.13,27.1-1.55c2.93-0.78,4.64-3.26,5.42-6.19C67.94,34.95,68,24,68,24S67.94,13.05,66.52,7.74z" fill="#f00"></path>
                    <path d="M 45,24 27,14 27,34" fill="#fff"></path>
                </svg>
            </div>
            <div class="video-time-badge">{int(start_time)}s - {int(end_time)}s</div>
        </div>
    </div>
    <div class="video-info">
        <h4 class="video-title">{video_title}</h4>
        <p class="video-meta">
            <span class="channel">{channel}</span>
            <span class="separator">•</span>
            <span class="date">{published_date}</span>
        </p>
    </div>
</div>
"""
            return embed_html
        else:
            # Subsequent occurrences: return a text link
            return f'<sup class="citation-link">[{citation_num}]</sup>'

    # Replace all citations
    result = re.sub(r'\[(\d+)\]', replace_citation, article)

    print(f"✓ Embedded {len(embedded_citations)} video clips")

    return result


def generate_html_newsletter(article_with_embeds: str, metadata: Dict) -> str:
    """
    Generate complete HTML newsletter with styling.

    Args:
        article_with_embeds: Article content with video embeds
        metadata: Newsletter metadata (title, date, keywords, etc.)

    Returns:
        Complete HTML document
    """
    # Convert markdown-style headers to HTML
    article_html = article_with_embeds

    # Convert # Header to <h1>
    article_html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', article_html, flags=re.MULTILINE)

    # Convert ## Header to <h2>
    article_html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', article_html, flags=re.MULTILINE)

    # Convert paragraphs (text between blank lines or headers)
    paragraphs = []
    current_para = []

    for line in article_html.split('\n'):
        line = line.strip()

        # Skip empty lines
        if not line:
            if current_para:
                para_text = ' '.join(current_para)
                # Don't wrap if it's already HTML (h1, h2, div, etc.)
                if not para_text.startswith('<'):
                    paragraphs.append(f'<p>{para_text}</p>')
                else:
                    paragraphs.append(para_text)
                current_para = []
        # Don't modify lines that are already HTML
        elif line.startswith('<'):
            if current_para:
                para_text = ' '.join(current_para)
                paragraphs.append(f'<p>{para_text}</p>')
                current_para = []
            paragraphs.append(line)
        else:
            current_para.append(line)

    # Handle last paragraph
    if current_para:
        para_text = ' '.join(current_para)
        if not para_text.startswith('<'):
            paragraphs.append(f'<p>{para_text}</p>')
        else:
            paragraphs.append(para_text)

    article_html = '\n'.join(paragraphs)

    # Generate keyword tags
    keyword_tags = ''.join([f'<span class="tag">{kw}</span>' for kw in metadata['keywords']])

    # Complete HTML template with modern design
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{metadata['title']}</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.7;
            color: #1a1a1a;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }}

        .container {{
            max-width: 900px;
            margin: 0 auto;
            background-color: #ffffff;
            border-radius: 20px;
            overflow: hidden;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }}

        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 60px 40px;
            text-align: center;
            position: relative;
            overflow: hidden;
        }}

        .header::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1440 320"><path fill="rgba(255,255,255,0.1)" d="M0,96L48,112C96,128,192,160,288,160C384,160,480,128,576,122.7C672,117,768,139,864,144C960,149,1056,139,1152,122.7C1248,107,1344,85,1392,74.7L1440,64L1440,320L1392,320C1344,320,1248,320,1152,320C1056,320,960,320,864,320C768,320,672,320,576,320C480,320,384,320,288,320C192,320,96,320,48,320L0,320Z"></path></svg>') no-repeat bottom;
            background-size: cover;
            opacity: 0.3;
        }}

        .header-content {{
            position: relative;
            z-index: 1;
        }}

        .header h1 {{
            font-size: 2.5em;
            font-weight: 800;
            margin-bottom: 15px;
            letter-spacing: -0.5px;
            text-shadow: 0 2px 10px rgba(0,0,0,0.2);
        }}

        .header .meta {{
            font-size: 1em;
            opacity: 0.95;
            font-weight: 300;
            letter-spacing: 0.5px;
        }}

        .tags {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            justify-content: center;
            margin-top: 25px;
        }}

        .tag {{
            background-color: rgba(255,255,255,0.25);
            backdrop-filter: blur(10px);
            padding: 8px 18px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: 500;
            border: 1px solid rgba(255,255,255,0.3);
            transition: all 0.3s ease;
        }}

        .tag:hover {{
            background-color: rgba(255,255,255,0.35);
            transform: translateY(-2px);
        }}

        .stats {{
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-top: 25px;
            flex-wrap: wrap;
        }}

        .stat {{
            background-color: rgba(255,255,255,0.2);
            backdrop-filter: blur(10px);
            padding: 12px 24px;
            border-radius: 12px;
            font-weight: 600;
            border: 1px solid rgba(255,255,255,0.3);
            transition: all 0.3s ease;
        }}

        .stat:hover {{
            background-color: rgba(255,255,255,0.3);
            transform: scale(1.05);
        }}

        .content {{
            padding: 60px 50px;
        }}

        .content h1 {{
            color: #1a1a1a;
            font-size: 2.8em;
            font-weight: 800;
            margin: 40px 0 25px 0;
            line-height: 1.2;
            letter-spacing: -1px;
        }}

        .content h2 {{
            color: #2d3748;
            font-size: 2em;
            font-weight: 700;
            margin: 50px 0 20px 0;
            padding-bottom: 15px;
            border-bottom: 3px solid #667eea;
            letter-spacing: -0.5px;
            position: relative;
        }}

        .content h2::after {{
            content: '';
            position: absolute;
            bottom: -3px;
            left: 0;
            width: 80px;
            height: 3px;
            background: linear-gradient(90deg, #764ba2, #667eea);
        }}

        .content p {{
            margin-bottom: 24px;
            font-size: 1.1em;
            line-height: 1.8;
            color: #4a5568;
        }}

        .citation-link {{
            color: #667eea;
            font-weight: 600;
            text-decoration: none;
            margin: 0 2px;
            transition: all 0.2s ease;
        }}

        .citation-link:hover {{
            color: #764ba2;
        }}

        .video-clip {{
            margin: 40px 0;
            border-radius: 16px;
            overflow: hidden;
            background: linear-gradient(135deg, #f6f8fb 0%, #ffffff 100%);
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
            border: 1px solid #e2e8f0;
        }}

        .video-clip:hover {{
            box-shadow: 0 15px 40px rgba(102,126,234,0.2);
            transform: translateY(-5px);
        }}

        .video-container {{
            position: relative;
            cursor: pointer;
        }}

        .video-thumbnail {{
            position: relative;
            padding-bottom: 56.25%;
            background-color: #000;
            background-size: cover;
            background-position: center;
            overflow: hidden;
            transition: all 0.3s ease;
        }}

        .video-container:hover .video-thumbnail {{
            transform: scale(1.02);
        }}

        .video-container iframe {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
        }}

        .play-button {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            transition: all 0.3s ease;
            filter: drop-shadow(0 4px 10px rgba(0,0,0,0.3));
            cursor: pointer;
            z-index: 10;
        }}

        .video-container:hover .play-button {{
            transform: translate(-50%, -50%) scale(1.1);
        }}

        .video-time-badge {{
            position: absolute;
            bottom: 12px;
            right: 12px;
            background: rgba(0,0,0,0.85);
            backdrop-filter: blur(10px);
            color: white;
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 0.85em;
            font-weight: 600;
            letter-spacing: 0.5px;
        }}

        .video-info {{
            padding: 20px 24px;
            background: white;
        }}

        .video-title {{
            font-size: 1.15em;
            font-weight: 600;
            color: #1a202c;
            margin-bottom: 8px;
            line-height: 1.4;
        }}

        .video-meta {{
            font-size: 0.9em;
            color: #718096;
            display: flex;
            align-items: center;
            gap: 8px;
            flex-wrap: wrap;
        }}

        .channel {{
            font-weight: 500;
            color: #667eea;
        }}

        .separator {{
            color: #cbd5e0;
        }}

        .footer {{
            background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%);
            color: white;
            padding: 50px 40px;
            text-align: center;
        }}

        .footer p {{
            margin: 10px 0;
            font-size: 0.95em;
        }}

        .footer strong {{
            font-size: 1.3em;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}

        .footer-note {{
            margin-top: 20px;
            opacity: 0.7;
            font-size: 0.85em;
            font-weight: 300;
        }}

        @media (max-width: 768px) {{
            body {{
                padding: 10px;
            }}

            .container {{
                border-radius: 15px;
            }}

            .header {{
                padding: 40px 25px;
            }}

            .header h1 {{
                font-size: 1.8em;
            }}

            .content {{
                padding: 40px 25px;
            }}

            .content h1 {{
                font-size: 2em;
            }}

            .content h2 {{
                font-size: 1.5em;
            }}

            .content p {{
                font-size: 1em;
            }}

            .stats {{
                gap: 15px;
            }}

            .stat {{
                padding: 10px 18px;
                font-size: 0.9em;
            }}

            .video-info {{
                padding: 16px 20px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="header-content">
                <h1>{metadata['title']}</h1>
                <div class="meta">Generated on {metadata['date']}</div>
                <div class="tags">
                    {keyword_tags}
                </div>
                <div class="stats">
                    <div class="stat">📹 {metadata['total_videos']} Videos</div>
                    <div class="stat">📝 {metadata['total_chunks']} Clips</div>
                </div>
            </div>
        </div>

        <div class="content">
            {article_html}
        </div>

        <div class="footer">
            <p><strong>AI-Powered Newsletter Generator</strong></p>
            <p>Powered by ChromaDB + Google Gemini AI</p>
            <p class="footer-note">
                This newsletter was automatically generated from YouTube video transcripts using advanced AI technology.
            </p>
        </div>
    </div>

    <script>
        function loadVideo(playButton) {{
            // Get the video clip container
            const videoClip = playButton.closest('.video-clip');
            const videoContainer = videoClip.querySelector('.video-container');
            const videoId = videoClip.getAttribute('data-video-id');
            const startTime = videoClip.getAttribute('data-start-time');

            // Create iframe element
            const iframe = document.createElement('iframe');
            iframe.setAttribute('width', '100%');
            iframe.setAttribute('height', '100%');
            iframe.setAttribute('src', `https://www.youtube.com/embed/${{videoId}}?start=${{startTime}}&autoplay=1`);
            iframe.setAttribute('title', 'YouTube video player');
            iframe.setAttribute('frameborder', '0');
            iframe.setAttribute('allow', 'accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share');
            iframe.setAttribute('allowfullscreen', '');
            iframe.style.position = 'absolute';
            iframe.style.top = '0';
            iframe.style.left = '0';

            // Replace thumbnail with iframe
            videoContainer.innerHTML = '';
            videoContainer.style.position = 'relative';
            videoContainer.style.paddingBottom = '56.25%';
            videoContainer.style.height = '0';
            videoContainer.appendChild(iframe);
        }}
    </script>
</body>
</html>"""

    return html


def format_article_for_whatsapp(article: str, chunks: List[Dict], keywords: List[str]) -> str:
    """
    Format article for WhatsApp with proper formatting and video links.

    Args:
        article: Article text with [1], [2], [3] citations
        chunks: List of transcript chunks
        keywords: Search keywords

    Returns:
        WhatsApp-formatted text with clickable video links
    """
    # Track which citations have been replaced
    replaced_citations = set()

    def replace_citation_with_link(match):
        citation_num = int(match.group(1))

        # Check if valid citation number
        if citation_num < 1 or citation_num > len(chunks):
            return match.group(0)

        chunk = chunks[citation_num - 1]
        metadata = chunk['metadata']

        video_id = metadata.get('video_id', '')
        video_url = metadata.get('video_url', '')
        video_title = metadata.get('video_title', 'Video')
        channel = metadata.get('channel', 'Unknown')
        start_time = metadata.get('chunk_start_time', 0)
        end_time = metadata.get('chunk_end_time', 0)

        # Create timestamped YouTube URL
        if video_url and '?' in video_url:
            timestamp_url = f"{video_url}&t={int(start_time)}s"
        else:
            timestamp_url = f"https://www.youtube.com/watch?v={video_id}&t={int(start_time)}s"

        # Only create detailed link for first occurrence
        if citation_num not in replaced_citations:
            replaced_citations.add(citation_num)
            # Clean video embed with title and link
            return f"\n\n🎬 *{video_title}*\n_{channel}_ • {int(start_time)}s-{int(end_time)}s\n{timestamp_url}\n"
        else:
            # Subsequent occurrences - just the citation number
            return f" [{citation_num}]"

    # Replace citations with video links
    formatted_text = re.sub(r'\[(\d+)\]', replace_citation_with_link, article)

    # Convert markdown headers to WhatsApp formatting
    # # Header -> *HEADER* (bold)
    formatted_text = re.sub(r'^# (.+)$', lambda m: f"\n*{m.group(1)}*\n{'━'*30}\n", formatted_text, flags=re.MULTILINE)

    # ## Header -> *Header* (bold with spacing)
    formatted_text = re.sub(r'^## (.+)$', lambda m: f"\n\n*{m.group(1)}*\n", formatted_text, flags=re.MULTILINE)

    # Clean up excessive newlines
    formatted_text = re.sub(r'\n{4,}', '\n\n\n', formatted_text)

    # Remove leading/trailing whitespace
    formatted_text = formatted_text.strip()

    return formatted_text


def split_message_intelligently(message: str, max_length: int = 1500) -> List[str]:
    """
    Split message into chunks intelligently by sections.

    Args:
        message: The full message text
        max_length: Maximum characters per chunk (default 1500 to leave room for headers)

    Returns:
        List of message chunks
    """
    # Split by major sections (lines with ===)
    sections = []
    current_section = []

    for line in message.split('\n'):
        if '===' in line or '━━━' in line:
            # This is a section divider
            if current_section:
                sections.append('\n'.join(current_section))
                current_section = []
            current_section.append(line)
        else:
            current_section.append(line)

    # Add last section
    if current_section:
        sections.append('\n'.join(current_section))

    # Now group sections into chunks
    chunks = []
    current_chunk = ""

    for section in sections:
        # If adding this section exceeds limit, start new chunk
        if current_chunk and len(current_chunk) + len(section) + 2 > max_length:
            chunks.append(current_chunk.strip())
            current_chunk = section + "\n\n"
        else:
            current_chunk += section + "\n\n"

    # Add final chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    # If any chunk is still too long, split by paragraphs
    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= max_length:
            final_chunks.append(chunk)
        else:
            # Split by paragraphs
            paragraphs = chunk.split('\n\n')
            temp_chunk = ""

            for para in paragraphs:
                if len(temp_chunk) + len(para) + 2 <= max_length:
                    temp_chunk += para + "\n\n"
                else:
                    if temp_chunk.strip():
                        final_chunks.append(temp_chunk.strip())
                    temp_chunk = para + "\n\n"

            if temp_chunk.strip():
                final_chunks.append(temp_chunk.strip())

    return final_chunks


def send_whatsapp_message(message: str, to_number: str, media_urls: List[str] = None) -> bool:
    """
    Send WhatsApp message using Twilio API with intelligent splitting.

    Args:
        message: The formatted message text
        to_number: Recipient's WhatsApp number (format: whatsapp:+1234567890)
        media_urls: Optional list of media URLs to send

    Returns:
        True if sent successfully, False otherwise
    """
    # Get Twilio credentials from environment
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    from_number = os.getenv("TWILIO_WHATSAPP_NUMBER")  # Format: whatsapp:+14155238886

    if not all([account_sid, auth_token, from_number]):
        raise ValueError(
            "Missing Twilio credentials. Please add to .env file:\n"
            "TWILIO_ACCOUNT_SID=your_account_sid\n"
            "TWILIO_AUTH_TOKEN=your_auth_token\n"
            "TWILIO_WHATSAPP_NUMBER=whatsapp:+14155238886"
        )

    # Ensure to_number has whatsapp: prefix
    if not to_number.startswith('whatsapp:'):
        to_number = f'whatsapp:{to_number}'

    try:
        client = Client(account_sid, auth_token)

        # Split message intelligently
        max_length = 1500  # Leave room for part headers
        chunks = split_message_intelligently(message, max_length)

        total_parts = len(chunks)
        print(f"\n📊 Message will be split into {total_parts} part(s)")

        messages_sent = 0

        for i, chunk in enumerate(chunks, 1):
            # Add part header if multiple parts
            if total_parts > 1:
                part_header = f"📬 *Part {i}/{total_parts}*\n{'='*25}\n\n"
                message_body = part_header + chunk
            else:
                message_body = chunk

            # Send message
            msg = client.messages.create(
                body=message_body,
                from_=from_number,
                to=to_number,
                media_url=media_urls if (media_urls and i == 1) else None  # Only send media with first part
            )

            print(f"✓ Part {i}/{total_parts} sent: {msg.sid} ({len(message_body)} chars)")
            messages_sent += 1

            # Small delay between messages to avoid rate limiting
            if i < total_parts:
                time.sleep(1)

        print(f"\n✅ Successfully sent {messages_sent} WhatsApp message(s)")
        return True

    except Exception as e:
        print(f"\n❌ Error sending WhatsApp message: {str(e)}")
        return False


def main():
    """Main orchestration function."""
    print("="*80)
    print("=== AI-Powered Newsletter Generator ===")
    print("="*80)
    print()

    # 1. Get user inputs
    keywords_input = input("Enter search keywords (comma-separated): ")
    keywords = [k.strip() for k in keywords_input.split(",") if k.strip()]

    if not keywords:
        print("Error: Please provide at least one keyword")
        return

    language = input("Enter target language (e.g., English, Hindi, Tamil): ").strip()
    if not language:
        language = "English"

    # Ask for output format
    print("\nOutput format:")
    print("  1. HTML newsletter (for web/email)")
    print("  2. WhatsApp message (via Twilio)")
    print("  3. Both (HTML + WhatsApp)")
    output_choice = input("Choose output format (1/2/3, default: 1): ").strip()

    if output_choice not in ['1', '2', '3', '']:
        output_choice = '1'

    html_output = output_choice in ['1', '3', '']
    whatsapp_output = output_choice in ['2', '3']

    # Get output filename if HTML is selected
    output_file = None
    if html_output:
        output_file = input("Enter HTML output filename (default: newsletter.html): ").strip()
        if not output_file:
            output_file = "newsletter.html"
        elif not output_file.endswith('.html'):
            output_file += '.html'

    # Get WhatsApp details if selected
    whatsapp_number = None
    if whatsapp_output:

        load_dotenv()
        default_whatsapp_number = os.getenv("SENDER_NUMBER", "").strip()
        whatsapp_number = input(f"Enter recipient WhatsApp number (e.g., +1234567890) [default: {default_whatsapp_number}]: ").strip()
        if not whatsapp_number:
            whatsapp_number = default_whatsapp_number
        if not whatsapp_number:
            print("Error: WhatsApp number is required")
            return

    print("\n" + "="*80)

    try:
        # 2. Search ChromaDB
        print(f"\n📊 Searching for: {', '.join(keywords)}")
        chunks = aggregate_search_results(keywords, n_results_per_keyword=10)

        if not chunks:
            print("No results found. Try different keywords.")
            return

        print(f"✓ Found {len(chunks)} unique transcript chunks")

        # 3. Generate article with Gemini
        print(f"\n✍️  Generating article in {language}...")
        gemini_model = setup_gemini_client()
        prompt = build_article_prompt(chunks, language, keywords)
        article = generate_article_with_gemini(gemini_model, prompt)

        # 4. Generate outputs based on user selection
        if html_output:
            print("\n🎬 Embedding video clips for HTML...")
            article_with_videos = replace_citations_with_video_clips(article, chunks)

            print("\n📄 Creating HTML newsletter...")
            html = generate_html_newsletter(
                article_with_videos,
                metadata={
                    'title': f"AI News Digest: {', '.join(keywords)}",
                    'date': datetime.now().strftime('%Y-%m-%d'),
                    'keywords': keywords,
                    'total_videos': len(set(c['metadata']['video_id'] for c in chunks)),
                    'total_chunks': len(chunks)
                }
            )

            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html)

            print(f"✓ HTML newsletter saved to: {output_file}")

        if whatsapp_output:
            print("\n📱 Formatting for WhatsApp...")
            whatsapp_message = format_article_for_whatsapp(article, chunks, keywords)

            # Save WhatsApp version to file for preview
            whatsapp_file = "newsletter_whatsapp.txt"
            with open(whatsapp_file, 'w', encoding='utf-8') as f:
                f.write(whatsapp_message)
            print(f"✓ WhatsApp message saved to: {whatsapp_file}")

            # Ask for confirmation before sending
            print(f"\n📊 Message stats:")
            print(f"   - Length: {len(whatsapp_message)} characters")
            print(f"   - Estimated messages: {(len(whatsapp_message) // 1600) + 1}")
            print(f"   - Recipient: {whatsapp_number}")

            # confirm = input("\nSend WhatsApp message now? (y/n): ").strip().lower()
            confirm = 'y'

            if confirm == 'y':
                print("\n📤 Sending WhatsApp message...")
                success = send_whatsapp_message(whatsapp_message, whatsapp_number)

                if not success:
                    print("\n⚠️  WhatsApp message not sent. Check your Twilio credentials in .env file.")
            else:
                print("\n⏸️  WhatsApp message not sent (user cancelled)")

        print("\n" + "="*80)
        print("✅ SUCCESS!")
        print("="*80)
        print(f"✓ Total sources: {len(chunks)} clips from {len(set(c['metadata']['video_id'] for c in chunks))} videos")
        print(f"✓ Language: {language}")

        if html_output:
            print(f"\n🌐 HTML: Open {output_file} in your browser")
        if whatsapp_output:
            print(f"📱 WhatsApp: Preview message in newsletter_whatsapp.txt")

        print("="*80)

    except Exception as e:
        print("\n" + "="*80)
        print("❌ ERROR")
        print("="*80)
        print(f"Error: {str(e)}")
        print("\nPlease check:")
        print("1. ChromaDB is set up (run: python chromadb_setup.py)")
        print("2. .env file contains GEMINI_API_KEY and GCP_PROJECT_ID")
        print("3. You have internet connection for Gemini API")
        print("="*80)


if __name__ == "__main__":
    main()
