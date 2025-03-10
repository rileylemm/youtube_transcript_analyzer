from flask import Flask, request, jsonify, render_template
from scripts.youtube_transcript import (
    get_transcript, get_video_title, save_video_metadata
)
from scripts.transcript_analyzer import TranscriptAnalyzer
from dotenv import load_dotenv
import os
import asyncio
import logging
import json
from datetime import datetime
import aiofiles
import urllib.parse
from typing import Optional, Dict

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'data')

# Initialize transcript analyzer with OpenAI API key
openai_api_key = os.getenv('OPENAI_API_KEY')
analyzer = TranscriptAnalyzer(openai_api_key=openai_api_key)

def sanitize_filename(name):
    """Convert a string to a safe filename."""
    # Replace spaces and special chars with underscores
    safe_name = "".join(c if c.isalnum() else '_' for c in name)
    # Remove consecutive underscores
    safe_name = '_'.join(filter(None, safe_name.split('_')))
    return safe_name.lower()

def get_timestamp_prefix():
    """Get current timestamp in YYYYMMDD_HHMMSS format."""
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def generate_transcript_filename(video_title):
    """Generate consistent filename for transcripts."""
    safe_title = sanitize_filename(video_title)
    return f"{get_timestamp_prefix()}__transcript__{safe_title}.json"

def generate_analysis_filename(video_title, analysis_type, model):
    """Generate consistent filename for analyses."""
    safe_title = sanitize_filename(video_title)
    safe_type = sanitize_filename(analysis_type)
    safe_model = sanitize_filename(model)
    return f"{get_timestamp_prefix()}__analysis__{safe_type}__{safe_model}__{safe_title}.json"

def generate_chat_filename(video_title):
    """Generate consistent filename for chat history."""
    safe_title = sanitize_filename(video_title)
    return f"{get_timestamp_prefix()}__chat__{safe_title}.json"

def find_latest_file(directory: str, pattern: str) -> Optional[str]:
    """Find the most recent file matching a pattern in a directory."""
    try:
        matching_files = [f for f in os.listdir(directory) if pattern in f]
        if not matching_files:
            return None
        latest_file = max(matching_files, key=lambda f: os.path.getmtime(os.path.join(directory, f)))
        return os.path.join(directory, latest_file)
    except Exception as e:
        logger.error(f"Error finding latest file: {str(e)}")
        return None

def load_json_file(filepath: str) -> Optional[Dict]:
    """Load and parse a JSON file."""
    try:
        if not filepath or not os.path.exists(filepath):
            return None
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON file: {str(e)}")
        return None

async def load_chat_history(video_title):
    """Load chat history for a specific video."""
    try:
        # Get sanitized video directory
        safe_title = sanitize_filename(video_title)
        video_dir = os.path.join(app.config['UPLOAD_FOLDER'], safe_title)
        
        if not os.path.exists(video_dir):
            return {"video_id": video_title, "messages": []}
        
        # Find the most recent chat history file
        chat_filename = find_latest_file(video_dir, '__chat__')
        
        if chat_filename:
            try:
                async with aiofiles.open(chat_filename, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    history = json.loads(content)
                    return history
            except Exception as e:
                logger.error(f"Error reading chat history file: {str(e)}", exc_info=True)
                return {"video_id": video_title, "messages": []}
        
        return {"video_id": video_title, "messages": []}
        
    except Exception as e:
        logger.error(f"Error loading chat history: {str(e)}", exc_info=True)
        return {"video_id": video_title, "messages": []}

async def save_chat_history(video_title, user_message, assistant_response):
    """Save chat history for a specific video."""
    try:
        # Get sanitized video directory
        safe_title = sanitize_filename(video_title)
        video_dir = os.path.join(app.config['UPLOAD_FOLDER'], safe_title)
        os.makedirs(video_dir, exist_ok=True)
        
        chat_filename = generate_chat_filename(video_title)
        chat_file = os.path.join(video_dir, chat_filename)
        
        # Load existing history
        history = await load_chat_history(video_title)
        
        # Add new messages
        timestamp = datetime.now().isoformat()
        history["messages"].extend([
            {
                "role": "user",
                "content": user_message,
                "timestamp": timestamp
            },
            {
                "role": "assistant",
                "content": assistant_response,
                "timestamp": timestamp
            }
        ])
        
        # Save updated history
        async with aiofiles.open(chat_file, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(history, indent=2, ensure_ascii=False))
            
    except Exception as e:
        logger.error(f"Error saving chat history: {str(e)}", exc_info=True)

async def get_video_context(video_title):
    """Get combined context from transcript and analyses."""
    try:
        # Get sanitized video directory
        safe_title = sanitize_filename(video_title)
        video_dir = os.path.join(app.config['UPLOAD_FOLDER'], safe_title)
        context = []
        
        # Add a focused system message
        context.append(
            "You are an expert AI assistant specializing in LLM technology and implementation. "
            "Your role is to help users understand and work with LLMs effectively. "
            "Focus on providing accurate, technical, and practical information about LLM concepts, "
            "implementation details, and best practices. "
            "If a question is not related to LLMs or AI, politely redirect the conversation to these topics."
        )
        
        # Get most recent technical summary first
        analysis_files = [f for f in os.listdir(video_dir) if '__analysis__technical_summary__' in f]
        if analysis_files:
            latest_summary = max(analysis_files, key=lambda f: os.path.getmtime(os.path.join(video_dir, f)))
            with open(os.path.join(video_dir, latest_summary), 'r', encoding='utf-8') as f:
                summary = json.load(f)
            context.append("\n### Latest Technical Summary:")
            context.append(summary['content'])
        
        # Get transcript segments
        transcript_file = find_latest_file(video_dir, '__transcript__')
        if transcript_file:
            with open(transcript_file, 'r', encoding='utf-8') as f:
                transcript = json.load(f)
            context.append("\n### Relevant Transcript Segments:")
            for seg in transcript:
                minutes = int(seg['start'] // 60)
                seconds = int(seg['start'] % 60)
                timestamp = f"[{minutes:02d}:{seconds:02d}]"
                # Only include segments that mention LLM-related keywords
                text = seg['text'].lower()
                if any(keyword in text for keyword in ['llm', 'ai', 'model', 'gpt', 'language model', 'prompt', 'token']):
                    context.append(f"{timestamp} {seg['text']}")
        
        # Get most recent analyses of other types
        for analysis_type in ['code_snippets', 'tools_and_resources', 'key_workflows']:
            files = [f for f in os.listdir(video_dir) if f'__analysis__{analysis_type}__' in f]
            if files:
                latest = max(files, key=lambda f: os.path.getmtime(os.path.join(video_dir, f)))
                with open(os.path.join(video_dir, latest), 'r', encoding='utf-8') as f:
                    analysis = json.load(f)
                context.append(f"\n### Latest {analysis_type.replace('_', ' ').title()}:")
                context.append(analysis['content'])
        
        # Join all context with proper spacing
        full_context = "\n".join(context)
        
        return full_context
    except Exception as e:
        logger.error(f"Error getting video context: {str(e)}", exc_info=True)
        return ""

@app.route('/')
def index():
    logger.debug("Serving index page")
    return render_template('index.html')

@app.route('/library')
def library():
    logger.debug("Serving library page")
    
    # Dictionary to store organized video data
    videos = {}
    
    try:
        # Scan the data directory
        for video_dir_name in os.listdir(app.config['UPLOAD_FOLDER']):
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_dir_name)
            
            # Skip if not a directory
            if not os.path.isdir(video_path):
                continue
                
            # Initialize video entry
            videos[video_dir_name] = {
                'video_dir': video_path,
                'original_transcript': None,
                'analyses': {
                    'technical_summary': {'mistral': [], 'gpt': []},
                    'full_context': {'mistral': [], 'gpt': []},
                    'code_snippets': {'mistral': [], 'gpt': []},
                    'tools_and_resources': {'mistral': [], 'gpt': []},
                    'key_workflows': {'mistral': [], 'gpt': []}
                }
            }
            
            # Find latest transcript file
            transcript_file = find_latest_file(video_path, '__transcript__')
            if transcript_file:
                videos[video_dir_name]['original_transcript'] = {
                    'filename': os.path.basename(transcript_file),
                    'timestamp': os.path.getmtime(transcript_file)
                }
            
            # Scan files in the video directory for analyses
            for filename in os.listdir(video_path):
                if '__analysis__' in filename and filename.endswith('.json'):
                    filepath = os.path.join(video_path, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            analysis_data = json.load(f)
                            
                        # Parse filename to get type and model
                        parts = filename.split('__')
                        if len(parts) >= 5:  # timestamp__analysis__type__model__title.json
                            analysis_type = parts[2]
                            model = parts[3]
                            timestamp = os.path.getmtime(filepath)
                            
                            if analysis_type in videos[video_dir_name]['analyses']:
                                videos[video_dir_name]['analyses'][analysis_type][model].append({
                                    'filename': filename,
                                    'timestamp': timestamp,
                                    'filepath': filepath
                                })
                                
                                # Sort analyses by timestamp (newest first)
                                videos[video_dir_name]['analyses'][analysis_type][model].sort(
                                    key=lambda x: x['timestamp'],
                                    reverse=True
                                )
                    except Exception as e:
                        logger.error(f"Error processing analysis file {filename}: {str(e)}")
                        continue
        
        # Sort videos by most recently modified analysis
        def get_latest_timestamp(video_data):
            timestamps = []
            
            # Check original transcript
            if video_data['original_transcript']:
                timestamps.append(video_data['original_transcript']['timestamp'])
            
            # Check all analyses
            for analysis_type in video_data['analyses']:
                for model in video_data['analyses'][analysis_type]:
                    for analysis in video_data['analyses'][analysis_type][model]:
                        timestamps.append(analysis['timestamp'])
            
            return max(timestamps) if timestamps else 0
        
        sorted_videos = dict(sorted(
            videos.items(),
            key=lambda x: get_latest_timestamp(x[1]),
            reverse=True
        ))
        
        return render_template('library.html', videos=sorted_videos)
        
    except Exception as e:
        logger.error(f"Error serving library page: {str(e)}", exc_info=True)
        return render_template('library.html', videos={}, error=str(e))

@app.route('/get_transcript', methods=['POST'])
async def get_video_transcript():
    try:
        logger.debug("Received /get_transcript request")
        data = request.get_json()
        video_url = data.get('video_url')
        
        if not video_url:
            logger.warning("No video URL provided in request")
            return jsonify({'error': 'No video URL provided'}), 400

        logger.info(f"Fetching transcript and metadata for video URL: {video_url}")
        try:
            transcript, metadata = await get_transcript(video_url)
            logger.info(f"Retrieved transcript with {len(transcript)} segments")
            logger.info(f"Retrieved metadata for video: {metadata['title']}")
        except Exception as e:
            logger.error(f"Error fetching transcript or metadata: {str(e)}", exc_info=True)
            return jsonify({'error': f'Failed to fetch video data: {str(e)}'}), 500

        # Create video directory using sanitized title
        video_title = metadata['title']
        safe_title = sanitize_filename(video_title)
        video_dir = os.path.join(app.config['UPLOAD_FOLDER'], safe_title)
        os.makedirs(video_dir, exist_ok=True)

        logger.info("Saving transcript and metadata")
        try:
            # Save transcript
            await analyzer.save_original_transcript(transcript, video_title)
            
            # Save metadata and thumbnails
            await save_video_metadata(metadata, video_dir)
            
            logger.info("Transcript and metadata saved successfully")
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}", exc_info=True)
            return jsonify({'error': f'Failed to save data: {str(e)}'}), 500

        # Format transcript with timestamps
        formatted_transcript = {
            'segments': [{
                'timestamp': f"[{int(seg['start'])//60:02d}:{int(seg['start'])%60:02d}]" if int(seg['start']) % 60 == 0 else None,
                'start': seg['start'],
                'text': seg['text'].strip()  # Remove any extra whitespace
            } for seg in transcript]
        }

        # Combine segments that are part of the same sentence
        combined_segments = []
        current_text = []
        current_timestamp = None
        last_minute = -1

        for segment in formatted_transcript['segments']:
            current_minute = int(segment['start']) // 60
            
            # If we've moved to a new minute and haven't stored the timestamp yet
            if current_minute > last_minute:
                # If we have accumulated text, add it to segments
                if current_text:
                    combined_segments.append({
                        'timestamp': current_timestamp,
                        'text': ' '.join(current_text)
                    })
                    current_text = []
                
                current_timestamp = f"[{current_minute:02d}:00]"
                last_minute = current_minute
            
            current_text.append(segment['text'])
        
        # Add any remaining text
        if current_text:
            combined_segments.append({
                'timestamp': current_timestamp or f"[{int(formatted_transcript['segments'][-1]['start'])//60:02d}:00]",
                'text': ' '.join(current_text)
            })

        # Update the response data with combined segments
        response_data = {
            'title': video_title,
            'transcript': {'segments': combined_segments},
            'metadata': {
                'channel_name': metadata['channel_name'],
                'channel_url': metadata['channel_url'],
                'publish_date': metadata['publish_date'],
                'view_count': metadata['view_count'],
                'description': metadata['description'],
                'thumbnail_url': metadata['thumbnail_url']
            }
        }

        logger.info("Successfully processed video data")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Unexpected error in get_video_transcript: {str(e)}", exc_info=True)
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

@app.route('/analyze_transcript', methods=['POST'])
async def analyze_transcript():
    try:
        logger.debug("Received /analyze_transcript request")
        data = request.get_json()
        video_title = data.get('video_title')
        analysis_type = data.get('analysis_type', 'technical_summary')
        model = data.get('model', 'mistral')
        
        logger.info(f"Analyzing transcript for video: {video_title}")
        logger.info(f"Analysis type: {analysis_type}, Model: {model}")
        
        if not video_title:
            logger.warning("No video title provided in request")
            return jsonify({'error': 'No video title provided'}), 400
            
        # Get sanitized video directory
        safe_title = sanitize_filename(video_title)
        video_dir = os.path.join(app.config['UPLOAD_FOLDER'], safe_title)
        
        if not os.path.exists(video_dir):
            logger.error(f"Video directory not found: {video_dir}")
            return jsonify({'error': 'Video not found'}), 404
            
        # Find the most recent transcript file
        transcript_file = find_latest_file(video_dir, '__transcript__')
        if not transcript_file:
            logger.error("No transcript file found")
            return jsonify({'error': 'Transcript not found'}), 404
            
        try:
            with open(transcript_file, 'r', encoding='utf-8') as f:
                transcript = json.load(f)
            logger.info("Retrieved original transcript")
        except Exception as e:
            logger.error(f"Error reading transcript file: {str(e)}", exc_info=True)
            return jsonify({'error': f'Failed to read transcript: {str(e)}'}), 500
            
        try:
            if model == 'mistral':
                logger.info("Analyzing with Mistral model")
                result = await analyzer.analyze_with_mistral(transcript, analysis_type)
            else:
                logger.info("Analyzing with GPT model")
                result = await analyzer.analyze_with_gpt(transcript, analysis_type)
            logger.info("Analysis completed successfully")
        except Exception as e:
            logger.error(f"Error during analysis: {str(e)}", exc_info=True)
            return jsonify({'error': f'Analysis failed: {str(e)}'}), 500
            
        try:
            await analyzer.save_analysis(result, video_title, analysis_type, model)
            logger.info("Analysis results saved successfully")
        except Exception as e:
            logger.error(f"Error saving analysis results: {str(e)}", exc_info=True)
            # Continue even if saving fails
            
        return jsonify({'analysis': result})
        
    except Exception as e:
        logger.error(f"Unexpected error in analyze_transcript: {str(e)}", exc_info=True)
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

@app.route('/get_saved_transcript', methods=['POST'])
async def get_saved_transcript():
    try:
        data = request.get_json()
        video_title = data.get('video_title')
        
        if not video_title:
            return jsonify({'error': 'No video title provided'}), 400
            
        # Get sanitized video directory
        safe_title = sanitize_filename(video_title)
        video_dir = os.path.join(app.config['UPLOAD_FOLDER'], safe_title)
        
        if not os.path.exists(video_dir):
            return jsonify({'error': 'Video not found'}), 404
            
        # Find the most recent transcript file
        transcript_file = find_latest_file(video_dir, '__transcript__')
        if not transcript_file:
            return jsonify({'error': 'Transcript not found'}), 404
            
        try:
            with open(transcript_file, 'r', encoding='utf-8') as f:
                transcript = json.load(f)
        except Exception as e:
            logger.error(f"Error reading transcript file: {str(e)}", exc_info=True)
            return jsonify({'error': f'Failed to read transcript: {str(e)}'}), 500
            
        # Format transcript with timestamps
        formatted_transcript = {
            'segments': []
        }
        
        last_minute = -1
        current_text = []
        current_timestamp = None
        
        for seg in transcript:
            current_minute = int(seg['start']) // 60
            
            # If we've moved to a new minute
            if current_minute > last_minute:
                # If we have accumulated text, add it to segments
                if current_text:
                    formatted_transcript['segments'].append({
                        'timestamp': current_timestamp,
                        'text': ' '.join(current_text)
                    })
                    current_text = []
                
                current_timestamp = f"[{current_minute:02d}:00]"
                last_minute = current_minute
            
            current_text.append(seg['text'])
        
        # Add any remaining text
        if current_text:
            formatted_transcript['segments'].append({
                'timestamp': current_timestamp or f"[{last_minute:02d}:00]",
                'text': ' '.join(current_text)
            })
        
        return jsonify(formatted_transcript)
        
    except Exception as e:
        logger.error(f"Error in get_saved_transcript: {str(e)}", exc_info=True)
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

@app.route('/get_analysis', methods=['POST'])
async def get_analysis():
    try:
        data = request.get_json()
        filepath = data.get('filepath')
        
        if not filepath:
            return jsonify({'error': 'No filepath provided'}), 400
            
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                analysis_data = json.load(f)
        except Exception as e:
            logger.error(f"Error reading analysis file: {str(e)}", exc_info=True)
            return jsonify({'error': f'Failed to read analysis: {str(e)}'}), 500
            
        return jsonify({'content': analysis_data['content']})
        
    except Exception as e:
        logger.error(f"Error in get_analysis: {str(e)}", exc_info=True)
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

# Add timestamp filter for Jinja templates
@app.template_filter('timestamp')
def timestamp_filter(value):
    """Convert a timestamp to a readable date format."""
    try:
        return datetime.fromtimestamp(value).strftime('%Y-%m-%d %H:%M:%S')
    except (TypeError, ValueError):
        return str(value)

@app.template_filter('urlencode')
def urlencode_filter(s):
    """URL encode a string."""
    if isinstance(s, str):
        s = s.encode('utf-8')
    return urllib.parse.quote_plus(s)

@app.route('/chat_with_video', methods=['POST'])
async def chat_with_video():
    try:
        data = request.get_json()
        video_title = data.get('video_title')
        message = data.get('message')
        
        if not video_title or not message:
            return jsonify({'error': 'Missing video title or message'}), 400
            
        # Get context and chat history
        context = await get_video_context(video_title)
        history = await load_chat_history(video_title)
        
        # Prepare system message with context
        system_message = (
            "You are a helpful AI assistant discussing a YouTube video. "
            "Below is the video's transcript and analyses. This information includes:\n"
            "1. The video title\n"
            "2. The full transcript with timestamps\n"
            "3. Various analyses of the content\n\n"
            "Use this information to answer questions about the video. "
            "Reference specific timestamps when relevant. "
            "If you can't find the answer in the provided context, say so clearly. "
            "Keep responses clear and concise.\n\n"
            f"{context}"
        )
        
        # Format chat history for Mistral
        chat_messages = [{"role": "system", "content": system_message}]
        # Add last few messages for context (limit to keep context window manageable)
        for msg in history["messages"][-6:]:  # Last 3 exchanges
            chat_messages.append({"role": msg["role"], "content": msg["content"]})
        # Add current message
        chat_messages.append({"role": "user", "content": message})
        
        try:
            # Get response from Mistral
            response = await analyzer.chat_with_mistral(chat_messages)
            
            # Save to history
            await save_chat_history(video_title, message, response)
            
            return jsonify({'response': response})
            
        except Exception as e:
            logger.error(f"Error getting Mistral response: {str(e)}", exc_info=True)
            return jsonify({'error': f'Failed to get response: {str(e)}'}), 500
            
    except Exception as e:
        logger.error(f"Error in chat_with_video: {str(e)}", exc_info=True)
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

@app.route('/get_chat_history', methods=['POST'])
async def get_chat_history():
    try:
        data = request.get_json()  # Remove await since Flask's request.get_json() is not async
        video_title = data.get('video_title')
        if not video_title:
            return jsonify({'error': 'Video title is required'}), 400
            
        # Get sanitized video directory
        safe_title = sanitize_filename(video_title)
        video_dir = os.path.join(app.config['UPLOAD_FOLDER'], safe_title)
        
        if not os.path.exists(video_dir):
            return jsonify({'video_id': video_title, 'messages': []}), 200
            
        # Find the most recent chat history file
        chat_filename = find_latest_file(video_dir, '__chat__')
        
        if chat_filename:
            try:
                with open(chat_filename, 'r', encoding='utf-8') as f:
                    history = json.load(f)
                return jsonify({'messages': history.get('messages', [])})
            except Exception as e:
                logger.error(f"Error reading chat history file: {str(e)}", exc_info=True)
                return jsonify({'error': f'Failed to read chat history: {str(e)}'}), 500
        else:
            return jsonify({'video_id': video_title, 'messages': []}), 200
            
    except Exception as e:
        logger.error(f"Error getting chat history: {str(e)}", exc_info=True)
        return jsonify({'error': f'Failed to load chat history: {str(e)}'}), 500

@app.route('/delete_chat_message', methods=['POST'])
async def delete_chat_message():
    try:
        data = request.get_json()  # Remove await since Flask's request.get_json() is not async
        video_title = data.get('video_title')
        timestamp = data.get('timestamp')
        
        if not video_title or not timestamp:
            return jsonify({'error': 'Video title and timestamp are required'}), 400
            
        # Load current history
        history = await load_chat_history(video_title)
        
        # Filter out the message with matching timestamp
        history['messages'] = [msg for msg in history.get('messages', []) if msg.get('timestamp') != timestamp]
        
        # Save updated history
        safe_title = sanitize_filename(video_title)
        video_dir = os.path.join(app.config['UPLOAD_FOLDER'], safe_title)
        chat_filename = generate_chat_filename(video_title)
        history_file = os.path.join(video_dir, chat_filename)
        
        async with aiofiles.open(history_file, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(history, indent=2, ensure_ascii=False))
            
        return jsonify({'success': True})
        
    except Exception as e:
        logger.error(f"Error deleting chat message: {str(e)}", exc_info=True)
        return jsonify({'error': 'Failed to delete message'}), 500

@app.route('/video/<video_title>')
async def video_page(video_title):
    try:
        video_dir = os.path.join(app.config['UPLOAD_FOLDER'], sanitize_filename(video_title))
        
        if not os.path.exists(video_dir):
            return render_template('error.html', error="Video not found"), 404

        # Get metadata for video URL
        metadata = None
        metadata_file = find_latest_file(video_dir, '__metadata__')
        if metadata_file:
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            except Exception as e:
                logger.error(f"Error reading metadata file: {str(e)}")

        # Get transcript - find most recent transcript file
        transcript = []
        transcript_filename = find_latest_file(video_dir, '__transcript__')
        if transcript_filename:
            transcript_path = os.path.join(video_dir, transcript_filename)
            with open(transcript_path, 'r', encoding='utf-8') as f:
                raw_transcript = json.load(f)
                
                # Process transcript with minute-based timestamps
                last_minute = -1
                current_text = []
                current_timestamp = None
                
                for seg in raw_transcript:
                    current_minute = int(seg['start']) // 60
                    
                    # If we've moved to a new minute
                    if current_minute > last_minute:
                        # If we have accumulated text, add it to transcript
                        if current_text:
                            transcript.append({
                                'timestamp': current_timestamp,
                                'text': ' '.join(current_text)
                            })
                            current_text = []
                        
                        current_timestamp = f"[{current_minute:02d}:00]"
                        last_minute = current_minute
                    
                    current_text.append(seg['text'])
                
                # Add any remaining text
                if current_text:
                    transcript.append({
                        'timestamp': current_timestamp or f"[{last_minute:02d}:00]",
                        'text': ' '.join(current_text)
                    })

        # Get analyses
        analyses = {
            'technical_summary': {'mistral': [], 'gpt': []},
            'full_context': {'mistral': [], 'gpt': []},
            'code_snippets': {'mistral': [], 'gpt': []},
            'tools_and_resources': {'mistral': [], 'gpt': []},
            'key_workflows': {'mistral': [], 'gpt': []}
        }
        
        # Group analysis files by type and model
        for filename in os.listdir(video_dir):
            if '__analysis__' in filename and filename.endswith('.json'):
                filepath = os.path.join(video_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        analysis_data = json.load(f)
                    
                    # Parse filename to get type and model
                    parts = filename.split('__')
                    if len(parts) >= 5:  # timestamp__analysis__type__model__title.json
                        analysis_type = parts[2]
                        model = parts[3]
                        timestamp = os.path.getmtime(filepath)
                        
                        if analysis_type in analyses:
                            analyses[analysis_type][model].append({
                                'content': analysis_data['content'],
                                'timestamp': timestamp
                            })
                except Exception as e:
                    logger.error(f"Error reading analysis file {filename}: {str(e)}")
                    continue

        # Sort analyses by timestamp
        for analysis_type in analyses:
            for model in analyses[analysis_type]:
                analyses[analysis_type][model].sort(key=lambda x: x['timestamp'], reverse=True)

        # Get chat history for stats
        chat_history = await load_chat_history(video_title)
        
        # Calculate stats
        stats = {
            'transcript_length': sum(len(seg['text'].split()) for seg in transcript),
            'analyses_count': sum(
                len(model_analyses) 
                for analysis_type in analyses.values() 
                for model_analyses in analysis_type.values()
            ),
            'chat_messages': len(chat_history.get('messages', [])),
        }
        
        # Calculate duration from transcript
        if transcript and raw_transcript:
            try:
                last_start = raw_transcript[-1]['start']
                minutes = int(last_start // 60)
                seconds = int(last_start % 60)
                stats['duration'] = f"{minutes}:{seconds:02d}"
            except (IndexError, ValueError, NameError):
                stats['duration'] = "0:00"
        else:
            stats['duration'] = "0:00"

        return render_template(
            'video_page.html',
            video_title=video_title,
            transcript=transcript,
            analyses=analyses,
            stats=stats,
            metadata=metadata  # Add metadata to template context
        )
        
    except Exception as e:
        logger.error(f"Error serving video page: {str(e)}", exc_info=True)
        return render_template('error.html', error=str(e)), 500

@app.context_processor
def utility_processor():
    """Add utility functions to template context."""
    return {
        'find_latest_file': find_latest_file,
        'load_json_file': load_json_file
    }

@app.route('/delete_video', methods=['POST'])
def delete_video():
    try:
        data = request.get_json()
        video_title = data.get('video_title')
        
        if not video_title:
            return jsonify({'error': 'Video title is required'}), 400
            
        # Get sanitized video directory
        safe_title = sanitize_filename(video_title)
        video_dir = os.path.join(app.config['UPLOAD_FOLDER'], safe_title)
        
        if not os.path.exists(video_dir):
            return jsonify({'error': 'Video not found'}), 404
            
        try:
            # Remove all files in the directory
            for filename in os.listdir(video_dir):
                file_path = os.path.join(video_dir, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    logger.error(f"Error deleting file {file_path}: {str(e)}")
            
            # Remove the directory
            os.rmdir(video_dir)
            
            return jsonify({'success': True})
            
        except Exception as e:
            logger.error(f"Error deleting video directory: {str(e)}")
            return jsonify({'error': f'Failed to delete video: {str(e)}'}), 500
            
    except Exception as e:
        logger.error(f"Error in delete_video: {str(e)}")
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

@app.route('/delete_analysis', methods=['POST'])
def delete_analysis():
    try:
        data = request.get_json()
        video_title = data.get('video_title')
        analysis_type = data.get('analysis_type')
        model = data.get('model')
        timestamp = data.get('timestamp')
        
        if not all([video_title, analysis_type, model, timestamp]):
            return jsonify({'error': 'Missing required parameters'}), 400
            
        # Get sanitized video directory
        safe_title = sanitize_filename(video_title)
        video_dir = os.path.join(app.config['UPLOAD_FOLDER'], safe_title)
        
        if not os.path.exists(video_dir):
            return jsonify({'error': 'Video not found'}), 404
            
        # Find the analysis file with matching timestamp
        target_file = None
        for filename in os.listdir(video_dir):
            if (f'__analysis__{analysis_type}__{model}__' in filename and 
                filename.endswith('.json')):
                filepath = os.path.join(video_dir, filename)
                if abs(os.path.getmtime(filepath) - float(timestamp)) < 1:  # Within 1 second
                    target_file = filepath
                    break
        
        if not target_file:
            return jsonify({'error': 'Analysis file not found'}), 404
            
        try:
            os.remove(target_file)
            return jsonify({'success': True})
        except Exception as e:
            logger.error(f"Error deleting analysis file: {str(e)}")
            return jsonify({'error': f'Failed to delete analysis: {str(e)}'}), 500
            
    except Exception as e:
        logger.error(f"Error in delete_analysis: {str(e)}")
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

if __name__ == '__main__':
    logger.info("Starting Flask application")
    app.run(debug=True, port=5002) 