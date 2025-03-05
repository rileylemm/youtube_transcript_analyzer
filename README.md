# YouTube Transcript Analyzer

A web application for fetching, analyzing, and extracting insights from YouTube video transcripts. Supports multiple language models (Mistral and GPT-4) for advanced analysis.

## Features

- Fetch transcripts from YouTube videos
- Clean and format transcript text
- Multiple analysis tools:
  - Technical Summary
  - Code & Commands Extraction
  - Tools & Resources List
  - Workflow Extraction
- Support for multiple LLM backends:
  - Mistral 7B (local via Ollama)
  - GPT-4 (via OpenAI API)
- Comprehensive Library Page:
  - View all analyzed videos in one place
  - Access saved transcripts and analyses
  - Run additional analyses directly from the library
  - Track analysis history with timestamps
  - Filter by model type (Mistral/GPT-4)

## Setup

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
- Copy `.env.example` to `.env`
- Set your OpenAI API key (optional)
- Configure Ollama settings if needed

4. Install and start Ollama:
- Follow instructions at [Ollama.ai](https://ollama.ai)
- Pull the Mistral model: `ollama pull mistral`

## Usage

1. Start the Flask server:
```bash
python app.py
```

2. Open your browser and navigate to `http://localhost:5000`

3. Paste a YouTube video URL and click "Get Transcript"

4. Use the analysis tools to extract insights:
- Choose between Mistral (local) or GPT-4
- Select the type of analysis you want to perform
- View the results in a structured format

5. Access the Library Page:
- Click "View Library" to see all analyzed videos
- Browse through saved transcripts and analyses
- Run additional analyses on any video
- Track analysis history with timestamps
- View results from different models side by side

## Project Structure

```
youtube_transcripts/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── scripts/
│   ├── youtube_transcript.py    # Transcript fetching and processing
│   └── transcript_analyzer.py   # Analysis logic for different models
├── static/
│   └── css/              # Stylesheets
└── templates/
    ├── index.html        # Main web interface
    └── library.html      # Library page interface
```

## Contributing

Feel free to open issues or submit pull requests for improvements.

## License

MIT License 