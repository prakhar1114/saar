import os
import datetime
import json
import isodate
import sys
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_channel_videos_yesterday(channel_name):
    """
    Fetches all YouTube videos published by a channel exactly one day before today (UTC).
    """
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise ValueError("Please set YOUTUBE_API_KEY in your .env file")

    youtube = build('youtube', 'v3', developerKey=api_key)

    try:
        uploads_playlist_id = None

        # 1. Handle Lookup (if starts with @)
        if channel_name.startswith('@'):
             print(f"Looking up channel handle: {channel_name}...")
             try:
                 handle_response = youtube.channels().list(
                    forHandle=channel_name,
                    part='contentDetails'
                ).execute()

                 if handle_response.get('items'):
                     uploads_playlist_id = handle_response['items'][0]['contentDetails']['relatedPlaylists']['uploads']
                     print(f"Found Channel via handle.")
                 else:
                     print(f"Handle '{channel_name}' not found, falling back to search...")
             except Exception as e:
                 print(f"Error looking up handle {channel_name}: {e}")

        # 2. Search Lookup (fallback or primary if no @)
        if not uploads_playlist_id:
             print(f"Searching for channel: {channel_name}...")
             search_response = youtube.search().list(
                q=channel_name,
                type='channel',
                part='id',
                maxResults=1
            ).execute()

             if not search_response['items']:
                print(f"Channel not found: {channel_name}")
                return []

             channel_id = search_response['items'][0]['id']['channelId']
             print(f"Found Channel ID: {channel_id}")

             # Get Uploads Playlist ID
             channel_response = youtube.channels().list(
                id=channel_id,
                part='contentDetails'
             ).execute()

             uploads_playlist_id = channel_response['items'][0]['contentDetails']['relatedPlaylists']['uploads']

        # 3. Calculate Yesterday's Date (UTC)
        today = datetime.datetime.utcnow().date()
        yesterday = today - datetime.timedelta(days=1)
        print(f"Looking for videos published on: {yesterday} (UTC)")

        # 4. Fetch Videos from Uploads Playlist
        videos = []

        # We might need to pagination if the channel uploads A LOT, but usually first page (50) is enough for 1 day.
        playlist_response = youtube.playlistItems().list(
            playlistId=uploads_playlist_id,
            part='snippet,contentDetails',
            maxResults=50
        ).execute()

        for item in playlist_response.get('items', []):
            published_at_str = item['contentDetails'].get('videoPublishedAt') or item['snippet'].get('publishedAt')

            # Parse ISO format (e.g., 2023-10-27T10:00:00Z)
            published_at = isodate.parse_datetime(published_at_str).replace(tzinfo=None) # naive UTC

            # Check if published yesterday
            if published_at.date() == yesterday:
                 # Extract video ID correctly
                 video_id = item['snippet']['resourceId']['videoId']
                 title = item['snippet']['title']
                 url = f"https://www.youtube.com/watch?v={video_id}"

                 videos.append({
                     'video_id': video_id,
                     'title': title,
                     'url': url,
                     'published_at': published_at_str,
                     'channel_name': channel_name
                 })

            # Since playlist items are usually ordered by date (newest first),
            # if we encounter a video OLDER than yesterday, we can stop.
            elif published_at.date() < yesterday:
                break

        print(f"Found {len(videos)} videos from yesterday for {channel_name}.")
        return videos

    except Exception as e:
        print(f"Error processing channel {channel_name}: {e}")
        return []

def get_video_transcript(video_id):
    """
    Fetches any available transcript for a given video ID.
    Returns a tuple: (transcript_data, metadata) or (None, None) if not available.
    """
    try:
        # Try to list all available transcripts first to show what's available
        ytt_api = YouTubeTranscriptApi()
        transcript_list = ytt_api.list(video_id)

        print(f"  Available transcripts for {video_id}:")

        # Show manually created transcripts
        manually_created = []
        try:
            for transcript in transcript_list:
                if not transcript.is_generated:
                    manually_created.append(f"{transcript.language_code} ({transcript.language})")
        except:
            pass

        print(f"    MANUALLY CREATED: {', '.join(manually_created) if manually_created else 'None'}")

        # Show generated transcripts
        generated = []
        try:
            for transcript in transcript_list:
                if transcript.is_generated:
                    generated.append(f"{transcript.language_code} ({transcript.language})")
        except:
            pass

        print(f"    GENERATED: {', '.join(generated) if generated else 'None'}")

        # Now fetch the first available transcript (prioritize manually created, fallback to auto-generated)
        selected_transcript = None
        try:
            # find_transcript automatically prioritizes manual over auto-generated
            # We'll try to get any available transcript by iterating through the list
            for transcript in transcript_list:
                selected_transcript = transcript
                break  # Take the first available one
        except:
            pass

        if selected_transcript:
            # Fetch the transcript
            fetched_data = selected_transcript.fetch()

            print(f"  ✓ Selected: {selected_transcript.language} ({selected_transcript.language_code}) - {'Auto-generated' if selected_transcript.is_generated else 'Manual'}")

            # Convert to formatted transcript
            # fetched_data is a FetchedTranscript object with a snippets property
            formatted_transcript = []
            for snippet in fetched_data.snippets:
                formatted_transcript.append({
                    'text': snippet.text,
                    'start_time': snippet.start,
                    'duration': snippet.duration
                })

            # Prepare metadata
            metadata = {
                'language': selected_transcript.language,
                'language_code': selected_transcript.language_code,
                'is_generated': selected_transcript.is_generated
            }

            return formatted_transcript, metadata
        else:
            print(f"  ✗ No transcript available")
            return None, None

    except TranscriptsDisabled:
        print(f"  ✗ Transcripts are disabled for this video")
        return None, None
    except NoTranscriptFound:
        print(f"  ✗ No transcript found for this video")
        return None, None
    except Exception as e:
        print(f"  ✗ Error fetching transcript: {e}")
        return None, None

def save_data_to_file(data, output_file='video_transcripts.json'):
    """Save data to JSON file."""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"  ✗ Error saving to file: {e}")
        return False

def save_chunked_transcripts(chunked_data, output_file='video_chunked_transcripts.jsonl'):
    """
    Save chunked transcript data to a JSONL file (one JSON object per line).
    This format is more efficient for large datasets and required by Vertex AI Search.

    Args:
        chunked_data: List of chunk records with metadata
        output_file: Output filename (should end in .jsonl)

    Returns:
        True if successful, False otherwise
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for chunk in chunked_data:
                f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
        return True
    except Exception as e:
        print(f"  ✗ Error saving chunked transcripts: {e}")
        return False

def chunk_transcript_by_time(transcript_data, chunk_duration=30):
    """
    Chunks transcript data into time-based segments.

    Args:
        transcript_data: List of transcript snippets with 'text', 'start_time', and 'duration'
        chunk_duration: Duration of each chunk in seconds (default: 30)

    Returns:
        List of chunks, where each chunk contains:
        - 'text': Combined text for the chunk
        - 'start_time': Start time of the chunk
        - 'end_time': End time of the chunk
        - 'snippets': List of original snippets in this chunk
    """
    if not transcript_data:
        return []

    chunks = []
    current_chunk_start = 0

    # Find the maximum end time to know when to stop
    max_end_time = max(
        snippet['start_time'] + snippet['duration']
        for snippet in transcript_data
    )

    # Create chunks for each time window
    while current_chunk_start <= max_end_time:
        chunk_end = current_chunk_start + chunk_duration

        # Collect all snippets that overlap with this time window
        snippets_in_chunk = []
        for snippet in transcript_data:
            snippet_start = snippet['start_time']
            snippet_end = snippet_start + snippet['duration']

            # Check if snippet overlaps with current chunk window
            # A snippet overlaps if it starts before chunk ends AND ends after chunk starts
            if snippet_start < chunk_end and snippet_end > current_chunk_start:
                snippets_in_chunk.append(snippet)

        # Only create a chunk if there are snippets in it
        if snippets_in_chunk:
            combined_text = ' '.join(s['text'] for s in snippets_in_chunk)

            chunk = {
                'text': combined_text,
                'start_time': current_chunk_start,
                'end_time': chunk_end,
                'snippets': snippets_in_chunk
            }
            chunks.append(chunk)

        current_chunk_start = chunk_end

    return chunks

def main():
    # ---------------------------------------------------------
    # CONFIGURATION: Add your channels here
    # ---------------------------------------------------------
    channels_to_scan = [
        "Zee Business",
        "@manoramanews",
        "@CNBC-TV18",
        "ET Now"
    ]
    # ---------------------------------------------------------

    all_data = []
    all_chunked_data = []  # New: Store flattened chunk records
    output_file = 'video_transcripts.json'
    chunked_output_file = 'video_chunked_transcripts.jsonl'
    print(f"--- Starting Job ---")

    for channel in channels_to_scan:
        print(f"\nProcessing: {channel}")

        # 1. Get videos
        videos = get_channel_videos_yesterday(channel)

        for video in videos[0:3]:
            print(f"\n  Processing: {video['title']}")

            # 2. Get transcript with metadata
            transcript_data, transcript_metadata = get_video_transcript(video['video_id'])

            # 3. Post-process: Chunk transcript into 60-second intervals
            transcript_chunks = None
            if transcript_data:
                transcript_chunks = chunk_transcript_by_time(transcript_data, chunk_duration=60)
                print(f"  ✓ Created {len(transcript_chunks)} chunks of 60 seconds each")

                # Create flattened chunk records with complete metadata
                for chunk in transcript_chunks:
                    chunk_record = {
                        # Video metadata
                        'channel': channel,
                        'video_title': video['title'],
                        'video_url': video['url'],
                        'video_id': video['video_id'],
                        'published_at': video['published_at'],

                        # Chunk timing
                        'chunk_start_time': chunk['start_time'],
                        'chunk_end_time': chunk['end_time'],
                        'chunk_duration': chunk['end_time'] - chunk['start_time'],

                        # Chunk content
                        'text': chunk['text'],

                        # Language metadata
                        'text_language': transcript_metadata.get('language') if transcript_metadata else None,
                        'text_language_code': transcript_metadata.get('language_code') if transcript_metadata else None,
                        'is_auto_generated': transcript_metadata.get('is_generated') if transcript_metadata else None,

                        # Additional metadata for search/indexing
                        'snippet_count': len(chunk['snippets']),  # Number of original snippets in this chunk
                        'video_published_date': video['published_at'].split('T')[0] if 'T' in video['published_at'] else video['published_at'],
                    }
                    all_chunked_data.append(chunk_record)

            video_record = {
                'channel': channel,
                'video_title': video['title'],
                'video_url': video['url'],
                'video_id': video['video_id'],
                'published_at': video['published_at'],
                'transcript': transcript_data,
                'transcript_metadata': transcript_metadata,
                'transcript_chunks': transcript_chunks  # Add chunked transcript
            }

            all_data.append(video_record)

            # Save continuously after each video
            if save_data_to_file(all_data, output_file):
                print(f"  ✓ Saved progress ({len(all_data)} videos so far) to {output_file}")

            # Save chunked data continuously
            if all_chunked_data and save_chunked_transcripts(all_chunked_data, chunked_output_file):
                print(f"  ✓ Saved chunked transcripts ({len(all_chunked_data)} chunks so far) to {chunked_output_file}")

    # Final summary
    print(f"\n{'='*60}")
    print(f"Job Complete!")
    print(f"Total videos processed: {len(all_data)}")
    videos_with_transcripts = sum(1 for v in all_data if v['transcript'] is not None)
    print(f"Videos with transcripts: {videos_with_transcripts}")
    print(f"Total chunks created: {len(all_chunked_data)}")
    print(f"Final data saved to: {output_file}")
    print(f"Final chunked data saved to: {chunked_output_file}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
