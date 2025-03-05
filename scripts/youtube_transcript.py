#!/usr/bin/env python3
"""
YouTube Transcript Fetcher

This script fetches and saves transcripts from YouTube videos.
It handles various YouTube URL formats and saves the transcripts
with metadata in JSON format.
"""

import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
import re
import aiohttp
import asyncio
import argparse
import logging
import ssl
import certifi

# Configure logging
logger = logging.getLogger(__name__)

def extract_video_id(url: str) -> Optional[str]:
    """
    Extract the video ID from a YouTube URL.
    
    Args:
        url: YouTube video URL in various formats
        
    Returns:
        str: Video ID if found, None otherwise
    """
    logger.debug(f"Extracting video ID from URL: {url}")
    try:
        parsed_url = urlparse(url)
        if parsed_url.hostname in ('youtu.be', 'www.youtu.be'):
            video_id = parsed_url.path[1:]
        elif parsed_url.hostname in ('youtube.com', 'www.youtube.com'):
            if parsed_url.path == '/watch':
                video_id = parse_qs(parsed_url.query)['v'][0]
            else:
                return None
        else:
            return None
            
        logger.info(f"Successfully extracted video ID: {video_id}")
        return video_id
    except Exception as e:
        logger.error(f"Error extracting video ID: {str(e)}", exc_info=True)
        return None


def get_best_transcript(video_id: str, lang: str = 'en') -> List[Dict]:
    """
    Get the best available transcript for a video.
    Tries to get manual transcripts first, then falls back to auto-generated.
    Also attempts to get transcripts with formatting if available.
    
    Args:
        video_id: YouTube video ID
        lang: Language code (default: 'en')
        
    Returns:
        List of transcript segments
    """
    try:
        # Get list of available transcripts
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        try:
            # First try to get manual transcript in target language
            transcript = transcript_list.find_manually_created_transcript([lang])
        except:
            try:
                # Then try to get manual transcript in any language and translate
                transcript = transcript_list.find_manually_created_transcript()
                transcript = transcript.translate(lang)
            except:
                try:
                    # Fall back to auto-generated in target language
                    transcript = transcript_list.find_generated_transcript([lang])
                except:
                    # Last resort: auto-generated in any language and translate
                    transcript = transcript_list.find_generated_transcript()
                    transcript = transcript.translate(lang)
        
        return transcript.fetch()
    
    except Exception as e:
        print(f"Error getting best transcript: {str(e)}")
        # Fall back to basic transcript fetch
        return YouTubeTranscriptApi.get_transcript(video_id, languages=[lang])


def clean_and_format_text(text: str) -> str:
    """
    Clean and format transcript text.
    - Fixes common OCR/auto-transcription errors
    - Adds proper capitalization
    - Ensures proper spacing and punctuation
    
    Args:
        text: Raw transcript text
        
    Returns:
        Cleaned and formatted text
    """
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Fix common transcription errors
    text = text.replace(" i ", " I ")
    text = text.replace(" im ", " I'm ")
    text = text.replace(" ive ", " I've ")
    text = text.replace(" id ", " I'd ")
    text = text.replace(" ill ", " I'll ")
    
    # Capitalize first letter of sentences
    text = '. '.join(s.capitalize() for s in text.split('. '))
    
    # Ensure proper spacing after punctuation
    text = re.sub(r'([.!?]),', r'\1', text)
    text = re.sub(r'([.!?])([^\s])', r'\1 \2', text)
    
    # Remove spaces before punctuation
    text = re.sub(r'\s+([.!?,])', r'\1', text)
    
    return text.strip()


def group_transcript_segments(transcript_list: List[Dict]) -> List[Dict]:
    """
    Group transcript segments into larger, meaningful chunks.
    Uses natural breaks and ensures proper formatting.
    """
    grouped_segments = []
    current_group = {
        'text': '',
        'start': 0,
        'duration': 0
    }
    
    for i, segment in enumerate(transcript_list):
        if i == 0:
            current_group['text'] = clean_and_format_text(segment['text'])
            current_group['start'] = segment['start']
            current_group['duration'] = segment['duration']
            continue
            
        prev_segment = transcript_list[i - 1]
        pause_duration = segment['start'] - (prev_segment['start'] + prev_segment['duration'])
        
        # Improved grouping logic
        should_start_new_group = (
            pause_duration > 2.0 or
            (pause_duration > 1.0 and 
             any(prev_segment['text'].strip().endswith(p) for p in '.!?')) or
            len(current_group['text']) > 300 or
            segment.get('new_paragraph', False)  # Some transcripts include paragraph markers
        )
        
        if should_start_new_group and current_group['text']:
            current_group['text'] = clean_and_format_text(current_group['text'])
            grouped_segments.append(current_group.copy())
            current_group = {
                'text': clean_and_format_text(segment['text']),
                'start': segment['start'],
                'duration': segment['duration']
            }
        else:
            # Add appropriate spacing between segments
            if not current_group['text'].endswith(('.', '!', '?', ',')):
                current_group['text'] += ' '
            current_group['text'] += segment['text']
            current_group['duration'] = (segment['start'] + segment['duration']) - current_group['start']
    
    # Add the last group
    if current_group['text']:
        current_group['text'] = clean_and_format_text(current_group['text'])
        grouped_segments.append(current_group)
    
    return grouped_segments


async def get_video_title(video_url: str) -> str:
    """
    Get the title of a YouTube video using oEmbed.
    This method doesn't require an API key.
    
    Args:
        video_url: YouTube video URL
        
    Returns:
        str: Video title, or video ID if title cannot be fetched
    """
    logger.debug(f"Fetching title for video URL: {video_url}")
    video_id = extract_video_id(video_url)
    if not video_id:
        logger.warning("Could not extract video ID from URL")
        return "Unknown Video"
        
    try:
        # Use YouTube's oEmbed endpoint to get video information
        oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
        
        # Create SSL context with certifi certificates
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.get(oembed_url) as response:
                if response.status == 200:
                    data = await response.json()
                    title = data['title']
                    # Clean the title to make it filesystem-friendly
                    title = re.sub(r'[<>:"/\\|?*]', '', title)  # Remove invalid filename characters
                    title = re.sub(r'\s+', '_', title)          # Replace spaces with underscores
                    logger.info(f"Successfully fetched video title: {title}")
                    return title
                else:
                    logger.warning(f"Failed to fetch video title. Status: {response.status}")
                    return video_id
    except Exception as e:
        logger.error(f"Error fetching video title: {str(e)}", exc_info=True)
        return video_id


async def get_transcript(video_url: str, lang: str = 'en') -> List[Dict]:
    """
    Fetch and format transcript for a YouTube video.
    Gets the best available transcript and applies formatting improvements.
    """
    video_id = extract_video_id(video_url)
    if not video_id:
        raise ValueError("Invalid YouTube URL")
    
    try:
        # Get the best available transcript
        # Run the synchronous YouTubeTranscriptApi calls in a thread pool
        transcript_list = await asyncio.to_thread(get_best_transcript, video_id, lang)
        
        # Group and format the transcript segments
        grouped_transcript = group_transcript_segments(transcript_list)
        
        return grouped_transcript
        
    except Exception as e:
        print(f"Error fetching transcript: {str(e)}")
        raise


async def save_transcript(transcript_data: Dict, output_dir: str = 'transcripts') -> str:
    """
    Save transcript data to a JSON file.
    
    Args:
        transcript_data: Dictionary containing transcript and metadata
        output_dir: Directory to save the transcript file (default: 'transcripts')
    
    Returns:
        str: Path to the saved file
    """
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename using timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"transcript_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Save to file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(transcript_data, f, indent=2, ensure_ascii=False)
    
    return filepath


async def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='Fetch and save YouTube video transcripts')
    parser.add_argument('url', help='YouTube video URL')
    parser.add_argument('--lang', default='en', help='Language code (default: en)')
    args = parser.parse_args()
    
    try:
        # Get transcript
        transcript_data = await get_transcript(args.url)
        
        # Save transcript
        filepath = await save_transcript(transcript_data)
        print(f"\nTranscript saved to: {filepath}")
        
        # Print first few segments
        print("\nFirst few segments of transcript:")
        for segment in transcript_data[:3]:
            print(f"\n[{formatTime(segment['start'])}]")
            print(segment['text'])
            
    except Exception as e:
        print(f"\nError: {str(e)}")
        return 1
    
    return 0


def formatTime(seconds: float) -> str:
    """Format time in seconds to MM:SS format"""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"


if __name__ == '__main__':
    import sys
    sys.exit(asyncio.run(main())) 