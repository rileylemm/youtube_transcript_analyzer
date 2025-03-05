from flask import Flask, request, jsonify, render_template
from scripts.youtube_transcript import get_transcript, get_video_title
from scripts.transcript_analyzer import TranscriptAnalyzer
from dotenv import load_dotenv
import os
import asyncio
import logging
import json
from datetime import datetime

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Initialize transcript analyzer with OpenAI API key
openai_api_key = os.getenv('OPENAI_API_KEY')
analyzer = TranscriptAnalyzer(openai_api_key=openai_api_key)

def load_chat_history(video_title):
    """Load chat history for a specific video."""
    try:
        video_dir = analyzer._get_video_dir(video_title)
        chat_file = os.path.join(video_dir, f"{video_title}_chat_history.json")
        
        if os.path.exists(chat_file):
            with open(chat_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"video_id": video_title, "messages": []}
    except Exception as e:
        logger.error(f"Error loading chat history: {str(e)}", exc_info=True)
        return {"video_id": video_title, "messages": []}

def save_chat_history(video_title, user_message, assistant_response):
    """Save chat history for a specific video."""
    try:
        video_dir = analyzer._get_video_dir(video_title)
        chat_file = os.path.join(video_dir, f"{video_title}_chat_history.json")
        
        # Load existing history or create new
        history = load_chat_history(video_title)
        
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
        with open(chat_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
            
    except Exception as e:
        logger.error(f"Error saving chat history: {str(e)}", exc_info=True)

async def get_video_context(video_title):
    """Get combined context from transcript and analyses."""
    try:
        video_dir = analyzer._get_video_dir(video_title)
        context = []
        
        # Get transcript
        transcript_file = os.path.join(video_dir, f"{video_title}_transcript_original.json")
        if os.path.exists(transcript_file):
            with open(transcript_file, 'r', encoding='utf-8') as f:
                transcript = json.load(f)
            context.append("\n### Video Transcript:")
            for seg in transcript:
                minutes = int(seg['start'] // 60)
                seconds = int(seg['start'] % 60)
                timestamp = f"[{minutes:02d}:{seconds:02d}]"
                context.append(f"{timestamp} {seg['text']}")
        
        # Get analyses
        for filename in os.listdir(video_dir):
            if '_analysis_' in filename and filename.endswith('.json'):
                filepath = os.path.join(video_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        analysis = json.load(f)
                    context.append(f"\n### {analysis['type'].replace('_', ' ').title()} Analysis:")
                    context.append(analysis['content'])
                except Exception as e:
                    logger.error(f"Error reading analysis file {filename}: {str(e)}")
                    continue
        
        # Join all context with proper spacing
        full_context = "\n".join(context)
        
        # Add video title at the start
        return f"### Video Title: {video_title}\n\n{full_context}"
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
        for video_dir in os.listdir(analyzer.data_dir):
            video_path = os.path.join(analyzer.data_dir, video_dir)
            
            # Skip if not a directory
            if not os.path.isdir(video_path):
                continue
                
            # Initialize video entry
            videos[video_dir] = {
                'original_transcript': None,
                'analyses': {
                    'technical_summary': {'mistral': [], 'gpt': []},
                    'code_snippets': {'mistral': [], 'gpt': []},
                    'tools_and_resources': {'mistral': [], 'gpt': []},
                    'key_workflows': {'mistral': [], 'gpt': []}
                }
            }
            
            # Scan files in video directory
            for filename in os.listdir(video_path):
                filepath = os.path.join(video_path, filename)
                
                # Skip directories
                if os.path.isdir(filepath):
                    continue
                
                # Handle original transcript
                if filename.endswith('_transcript_original.json'):
                    videos[video_dir]['original_transcript'] = {
                        'filename': filename,
                        'timestamp': os.path.getmtime(filepath)
                    }
                    continue
                
                # Handle analysis files
                if '_analysis_' in filename and filename.endswith('.json'):
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            analysis_data = json.load(f)
                            
                        analysis_type = analysis_data.get('type')
                        model = analysis_data.get('model', 'mistral')  # Default to mistral if not specified
                        timestamp = os.path.getmtime(filepath)
                        
                        if analysis_type in videos[video_dir]['analyses']:
                            videos[video_dir]['analyses'][analysis_type][model].append({
                                'filename': filename,
                                'timestamp': timestamp,
                                'filepath': filepath
                            })
                            
                            # Sort analyses by timestamp (newest first)
                            videos[video_dir]['analyses'][analysis_type][model].sort(
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

        logger.info(f"Fetching title for video URL: {video_url}")
        try:
            video_title = await get_video_title(video_url)
            logger.info(f"Retrieved video title: {video_title}")
        except Exception as e:
            logger.error(f"Error fetching video title: {str(e)}", exc_info=True)
            return jsonify({'error': f'Failed to fetch video title: {str(e)}'}), 500

        logger.info("Fetching transcript")
        try:
            transcript = await get_transcript(video_url)
            logger.info(f"Retrieved transcript with {len(transcript)} segments")
        except Exception as e:
            logger.error(f"Error fetching transcript: {str(e)}", exc_info=True)
            return jsonify({'error': f'Failed to fetch transcript: {str(e)}'}), 500

        logger.info("Saving original transcript")
        try:
            await analyzer.save_original_transcript(transcript, video_title)
            logger.info("Original transcript saved successfully")
        except Exception as e:
            logger.error(f"Error saving original transcript: {str(e)}", exc_info=True)
            return jsonify({'error': f'Failed to save transcript: {str(e)}'}), 500

        # Format transcript with timestamps
        formatted_transcript = {
            'segments': [{
                'timestamp': f"[{int(seg['start'])//60:02d}:{int(seg['start'])%60:02d}]",
                'start': seg['start'],
                'text': seg['text']
            } for seg in transcript]
        }

        logger.info("Successfully processed video transcript")
        return jsonify({
            'title': video_title,
            'transcript': formatted_transcript
        })
        
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
            
        # Get the original transcript from the saved file
        video_dir = analyzer._get_video_dir(video_title)
        transcript_path = os.path.join(video_dir, f"{video_title}_transcript_original.json")
        
        try:
            with open(transcript_path, 'r', encoding='utf-8') as f:
                transcript = json.load(f)
            logger.info("Retrieved original transcript")
        except Exception as e:
            logger.error(f"Error retrieving original transcript: {str(e)}", exc_info=True)
            return jsonify({'error': f'Failed to retrieve transcript: {str(e)}'}), 500
            
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
            
        video_dir = analyzer._get_video_dir(video_title)
        transcript_path = os.path.join(video_dir, f"{video_title}_transcript_original.json")
        
        try:
            with open(transcript_path, 'r', encoding='utf-8') as f:
                transcript = json.load(f)
        except Exception as e:
            logger.error(f"Error reading transcript file: {str(e)}", exc_info=True)
            return jsonify({'error': f'Failed to read transcript: {str(e)}'}), 500
            
        # Format transcript with timestamps
        formatted_transcript = {
            'segments': [{
                'timestamp': f"[{int(seg['start'])//60:02d}:{int(seg['start'])%60:02d}]",
                'text': seg['text']
            } for seg in transcript]
        }
        
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
def format_timestamp(timestamp):
    dt = datetime.fromtimestamp(timestamp)
    return dt.strftime('%Y-%m-%d %H:%M:%S')

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
        history = load_chat_history(video_title)
        
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
            save_chat_history(video_title, message, response)
            
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
        data = request.get_json()
        video_title = data.get('video_title')
        
        if not video_title:
            return jsonify({'error': 'No video title provided'}), 400
            
        history = load_chat_history(video_title)
        return jsonify(history)
        
    except Exception as e:
        logger.error(f"Error getting chat history: {str(e)}", exc_info=True)
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

@app.route('/delete_chat_message', methods=['POST'])
async def delete_chat_message():
    try:
        data = request.get_json()
        video_title = data.get('video_title')
        timestamp = data.get('timestamp')
        
        if not video_title or not timestamp:
            return jsonify({'error': 'Missing video title or timestamp'}), 400
            
        video_dir = analyzer._get_video_dir(video_title)
        chat_file = os.path.join(video_dir, f"{video_title}_chat_history.json")
        
        try:
            with open(chat_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
            
            # Remove the message pair (user message and assistant response) with the given timestamp
            history['messages'] = [msg for msg in history['messages'] 
                                 if msg['timestamp'] != timestamp]
            
            with open(chat_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
                
            return jsonify({'success': True})
            
        except Exception as e:
            logger.error(f"Error deleting chat message: {str(e)}", exc_info=True)
            return jsonify({'error': f'Failed to delete message: {str(e)}'}), 500
            
    except Exception as e:
        logger.error(f"Error in delete_chat_message: {str(e)}", exc_info=True)
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

if __name__ == '__main__':
    logger.info("Starting Flask application")
    app.run(debug=True) 