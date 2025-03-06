"""
Transcript Analysis Module

Provides AI-powered analysis tools for extracting valuable information from YouTube transcripts.
Supports both local Ollama models and OpenAI's GPT models.
"""

import json
import requests
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import os
import asyncio
from pathlib import Path
from datetime import datetime
from openai import AsyncOpenAI
import aiohttp
import logging
import re

logger = logging.getLogger(__name__)

@dataclass
class AnalysisResult:
    """Container for analysis results"""
    success: bool
    content: Union[Dict, List, str]
    model_used: str
    error: Optional[str] = None

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

class TranscriptAnalyzer:
    def __init__(self, ollama_base_url: str = "http://localhost:11434", openai_api_key: Optional[str] = None):
        """
        Initialize the analyzer with optional OpenAI API key.
        
        Args:
            ollama_base_url: Base URL for Ollama models
            openai_api_key: Optional OpenAI API key for GPT models
        """
        self.ollama_base_url = ollama_base_url
        self.openai_api_key = openai_api_key
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        if openai_api_key:
            self.openai_client = AsyncOpenAI(
                api_key=openai_api_key,
                timeout=60.0  # Increase timeout for longer transcripts
            )
        else:
            self.openai_client = None
        
    def _get_video_dir(self, video_title: str) -> str:
        """Get or create directory for video data."""
        safe_title = sanitize_filename(video_title)
        video_dir = os.path.join(self.data_dir, safe_title)
        os.makedirs(video_dir, exist_ok=True)
        return video_dir

    async def save_original_transcript(self, transcript: List[Dict], video_title: str) -> str:
        """Save original transcript to JSON file."""
        try:
            video_dir = self._get_video_dir(video_title)
            filename = generate_transcript_filename(video_title)
            filepath = os.path.join(video_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(transcript, f, ensure_ascii=False, indent=2)
                
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving transcript: {str(e)}", exc_info=True)
            raise

    async def save_analysis(self, analysis_content: str, video_title: str, analysis_type: str, model: str = "mistral") -> str:
        """Save analysis results to JSON file."""
        try:
            video_dir = self._get_video_dir(video_title)
            filename = generate_analysis_filename(video_title, analysis_type, model)
            filepath = os.path.join(video_dir, filename)
            
            analysis_data = {
                'type': analysis_type,
                'model': model,
                'content': analysis_content,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(analysis_data, f, ensure_ascii=False, indent=2)
                
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving analysis: {str(e)}", exc_info=True)
            raise

    async def analyze_with_mistral(self, transcript: List[Dict], analysis_type: str) -> str:
        """Analyze the transcript using Mistral model."""
        # Convert transcript segments to text with timestamps
        full_text = ""
        for segment in transcript:
            minutes = int(segment['start'] // 60)
            seconds = int(segment['start'] % 60)
            timestamp = f"[{minutes:02d}:{seconds:02d}] "
            full_text += timestamp + segment['text'] + "\n\n"

        system_messages = {
            "technical_summary": (
                "You are an expert in AI and LLM technology analysis. Your task is to create a focused summary of the video's technical content about LLMs and AI, specifically addressing:\n"
                "1. Core LLM/AI concepts and techniques discussed\n"
                "2. Specific implementation details and best practices\n"
                "3. Unique insights about LLM usage and capabilities\n"
                "4. Practical applications and limitations mentioned\n"
                "5. Technical challenges and solutions presented\n\n"
                "Format the output with clear headings and concise bullet points.\n"
                "Focus only on the most valuable technical insights about LLM technology."
            ),
            "code_snippets": (
                "Extract and analyze code related to LLM implementation and usage. For each code snippet:\n"
                "1. Identify the specific LLM-related functionality (e.g., prompt engineering, API calls, etc.)\n"
                "2. Explain the technical implementation details\n"
                "3. Highlight best practices for LLM integration\n"
                "4. Note any performance considerations or limitations\n"
                "5. Suggest potential improvements or alternatives\n\n"
                "Focus on code that directly interfaces with or supports LLM functionality."
            ),
            "tools_and_resources": (
                "Identify all LLM-related tools, libraries, and resources mentioned. For each:\n"
                "1. Specific role in LLM implementation or usage\n"
                "2. Technical capabilities and limitations\n"
                "3. Integration requirements and compatibility\n"
                "4. Performance characteristics and scaling considerations\n"
                "5. Alternatives and trade-offs discussed\n\n"
                "Focus on tools that are directly relevant to LLM development and deployment."
            ),
            "key_workflows": (
                "Document workflows and processes specific to LLM implementation and usage:\n"
                "1. Step-by-step procedures for LLM integration\n"
                "2. Best practices for prompt engineering\n"
                "3. Testing and validation approaches\n"
                "4. Performance optimization techniques\n"
                "5. Error handling and edge cases\n\n"
                "Emphasize practical, actionable workflows for LLM development."
            )
        }
        
        system_msg = system_messages.get(analysis_type, system_messages["technical_summary"])
        prompt = f"{system_msg}\n\nTranscript:\n{full_text}\n\nAnalysis:"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.ollama_base_url}/api/generate",
                    json={
                        "model": "mistral",
                        "prompt": prompt,
                        "stream": False
                    }
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result['response'].strip()
                    else:
                        error_text = await response.text()
                        raise Exception(f"Mistral API error: {error_text}")
                        
        except Exception as e:
            raise Exception(f"Error analyzing with Mistral: {str(e)}")

    async def analyze_with_gpt(self, transcript: List[Dict], analysis_type: str) -> str:
        """Analyze the transcript using OpenAI's GPT model."""
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not provided")

        # Convert transcript segments to text with timestamps
        full_text = ""
        for segment in transcript:
            minutes = int(segment['start'] // 60)
            seconds = int(segment['start'] % 60)
            timestamp = f"[{minutes:02d}:{seconds:02d}] "
            full_text += timestamp + segment['text'] + "\n\n"
            
        client = AsyncOpenAI(
            api_key=self.openai_api_key,
            timeout=60.0
        )
        
        system_messages = {
            "technical_summary": (
                "You are an expert in AI and LLM technology analysis. Your task is to create a focused summary of the video's technical content about LLMs and AI, specifically addressing:\n"
                "1. Core LLM/AI concepts and techniques discussed\n"
                "2. Specific implementation details and best practices\n"
                "3. Unique insights about LLM usage and capabilities\n"
                "4. Practical applications and limitations mentioned\n"
                "5. Technical challenges and solutions presented\n\n"
                "Format the output with clear headings and concise bullet points.\n"
                "Focus only on the most valuable technical insights about LLM technology."
            ),
            "code_snippets": (
                "Extract and analyze code related to LLM implementation and usage. For each code snippet:\n"
                "1. Identify the specific LLM-related functionality (e.g., prompt engineering, API calls, etc.)\n"
                "2. Explain the technical implementation details\n"
                "3. Highlight best practices for LLM integration\n"
                "4. Note any performance considerations or limitations\n"
                "5. Suggest potential improvements or alternatives\n\n"
                "Focus on code that directly interfaces with or supports LLM functionality."
            ),
            "tools_and_resources": (
                "Identify all LLM-related tools, libraries, and resources mentioned. For each:\n"
                "1. Specific role in LLM implementation or usage\n"
                "2. Technical capabilities and limitations\n"
                "3. Integration requirements and compatibility\n"
                "4. Performance characteristics and scaling considerations\n"
                "5. Alternatives and trade-offs discussed\n\n"
                "Focus on tools that are directly relevant to LLM development and deployment."
            ),
            "key_workflows": (
                "Document workflows and processes specific to LLM implementation and usage:\n"
                "1. Step-by-step procedures for LLM integration\n"
                "2. Best practices for prompt engineering\n"
                "3. Testing and validation approaches\n"
                "4. Performance optimization techniques\n"
                "5. Error handling and edge cases\n\n"
                "Emphasize practical, actionable workflows for LLM development."
            )
        }
        
        system_msg = system_messages.get(analysis_type, system_messages["technical_summary"])
        
        try:
            response = await client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": full_text}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            raise Exception(f"Error analyzing with GPT: {str(e)}")

    async def analyze_transcript(self, 
                               transcript: List[Dict], 
                               analysis_type: str,
                               video_title: str,
                               model: str = "mistral") -> AnalysisResult:
        """
        Main analysis method that routes to appropriate model.
        
        Args:
            transcript: List of transcript segments
            analysis_type: Type of analysis to perform
            video_title: Title of the video
            model: Model to use ('mistral' or 'gpt')
        """
        try:
            # Save the original transcript first
            await self.save_original_transcript(transcript, video_title)
            
            # Get analysis based on selected model
            if model == "mistral":
                content = await self.analyze_with_mistral(transcript, analysis_type)
            elif model == "gpt":
                content = await self.analyze_with_gpt(transcript, analysis_type)
            else:
                return AnalysisResult(
                    success=False,
                    content="",
                    model_used=model,
                    error=f"Unknown model: {model}"
                )

            # Save the analysis result
            await self.save_analysis(content, video_title, analysis_type, model)
            
            return AnalysisResult(
                success=True,
                content=content,
                model_used=model
            )
            
        except Exception as e:
            logger.error(f"Error analyzing transcript: {str(e)}", exc_info=True)
            return AnalysisResult(
                success=False,
                content="",
                model_used=model,
                error=str(e)
            )

    async def chat_with_mistral(self, messages: List[Dict]) -> str:
        """Chat with Mistral model using provided messages."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.ollama_base_url}/api/chat",
                    json={
                        "model": "mistral-base",
                        "messages": messages,
                        "stream": False
                    }
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        if 'message' in result:
                            return result['message']['content']
                        else:
                            logger.error(f"Unexpected API response format: {result}")
                            raise Exception("Unexpected API response format")
                    else:
                        error_text = await response.text()
                        raise Exception(f"Mistral API error: {error_text}")
                        
        except Exception as e:
            logger.error(f"Error in chat_with_mistral: {str(e)}")
            raise Exception(f"Error chatting with Mistral: {str(e)}") 