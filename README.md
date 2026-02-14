# AI-Powered YouTube Newsletter Generator

This tool automates the process of tracking YouTube channels, processing their content, and generating high-quality AI newsletters. It fetches video transcripts, indexes them using semantic search (ChromaDB), and uses Google Gemini to write comprehensive digests with embedded video clips.

## 🚀 Features

- **Automated Monitoring**: Scans specified YouTube channels for videos published yesterday.
- **Smart Retrieval**: Fetches transcripts (manual or auto-generated) and chunks them for granular search.
- **Semantic Search**: Uses **ChromaDB** and **Google Vertex AI** embeddings to find relevant video clips based on your keywords.
- **AI Writing**: Uses **Google Gemini** to synthesize information from multiple videos into a cohesive news article.
- **Rich Output**:
  - **HTML Newsletter**: Beautifully styled with **embedded video clips** that start exactly at the right timestamp.
  - **WhatsApp**: Formatted messages with timestamped links, ready to send via Twilio.

## 🛠️ Prerequisites

- **Python 3.8+**
- **Google Cloud Platform Project** with Vertex AI API enabled.
- **API Keys**:
  - YouTube Data API v3
  - Google Gemini API
  - Twilio (optional, for WhatsApp)

## 📦 Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd saar
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure Environment Variables:**
    Create a `.env` file in the root directory with the following keys:

    ```ini
    # Google / YouTube
    YOUTUBE_API_KEY=your_youtube_api_key
    GEMINI_API_KEY=your_gemini_api_key
    GCP_PROJECT_ID=your_google_cloud_project_id

    # Twilio (Optional - for WhatsApp)
    TWILIO_ACCOUNT_SID=your_twilio_sid
    TWILIO_AUTH_TOKEN=your_twilio_auth_token
    TWILIO_WHATSAPP_NUMBER=whatsapp:+14155238886
    SENDER_NUMBER=+1234567890  # Default recipient number
    ```

## 🏃‍♂️ Usage Workflow

### Step 1: Fetch Video Data
Scans your configured channels for videos published yesterday and saves their transcripts.

1.  Open `fetch_youtube_data.py` and update the `channels_to_scan` list:
    ```python
    channels_to_scan = [
        "Marques Brownlee",
        "@Veritasium",
        "CNBC Television"
    ]
    ```
2.  Run the script:
    ```bash
    python fetch_youtube_data.py
    ```
    *Output: `video_transcripts.json` and `video_chunked_transcripts.jsonl`*

### Step 2: Build/Update Vector Database
Ingests the chunked transcripts into ChromaDB for semantic search.

```bash
python chromadb_setup.py
```
*Output: Creates/Updates `./chroma_db` directory.*

### Step 3: Generate Newsletter
Searches the database for your keywords and generates the newsletter.

```bash
python generate_newsletter.py
```

**Interactive Prompts:**
1.  **Keywords**: Enter topics to search for (e.g., "AI, Stock Market, Apple").
2.  **Language**: Target language for the article (e.g., "English", "Hindi").
3.  **Output Format**: HTML, WhatsApp, or Both.

**Outputs:**
- `newsletter.html`: Open in your browser to view the interactive article.
- `newsletter_whatsapp.txt`: Formatted text block for WhatsApp.
- *(Optional)* Sends WhatsApp message directly if Twilio is configured.

## 📂 Project Structure

- `fetch_youtube_data.py`: Fetches videos & transcripts.
- `chromadb_setup.py`: Indexes data into Vector DB.
- `generate_newsletter.py`: Main AI generation logic.
- `chromadb_search.py`: Helper tool to test search queries manually.
- `requirements.txt`: Python dependencies.
