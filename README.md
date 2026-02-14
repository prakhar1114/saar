# YouTube Video Transcript Fetcher

This tool fetches videos published **yesterday** from a list of YouTube channels and downloads their transcripts (including timestamps).

## Setup

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **API Key:**
    Ensure you have a `.env` file in this directory with your YouTube Data API key:
    ```
    YOUTUBE_API_KEY=your_api_key_here
    ```

## Usage

1.  **Open `fetch_youtube_data.py`** and edit the `channels_to_scan` list with the channel names you want to track.

    ```python
    channels_to_scan = [
        "Marques Brownlee",
        "Veritasium",
        # Add your channels here
    ]
    ```

2.  **Run the script:**
    ```bash
    python3 fetch_youtube_data.py
    ```

## Output

The script saves the data to `video_transcripts.json` in the following format:

```json
[
    {
        "channel": "Channel Name",
        "video_title": "Video Title",
        "video_url": "https://www.youtube.com/watch?v=...",
        "published_at": "2023-10-27T10:00:00Z",
        "transcript": [
             {"text": "Hello world", "start": 0.0, "duration": 1.5},
             ...
        ]
    }
]
```

## Functions

If you want to use the logic in your own scripts, you can import the functions:

*   `get_channel_videos_yesterday(channel_name)`: Returns a list of video objects.
*   `get_video_transcript(video_id)`: Returns the transcript for a video.
