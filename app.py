from flask import Flask, request, jsonify, render_template
from scripts.youtube_transcript import get_transcript, get_video_title
from scripts.transcript_analyzer import TranscriptAnalyzer
from dotenv import load_dotenv
import os
import asyncio
import logging
import json

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
    from datetime import datetime
    dt = datetime.fromtimestamp(timestamp)
    return dt.strftime('%Y-%m-%d %H:%M:%S')

if __name__ == '__main__':
    logger.info("Starting Flask application")
    app.run(debug=True) 