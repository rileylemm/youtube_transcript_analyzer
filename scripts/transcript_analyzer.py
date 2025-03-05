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
        self.data_dir = "data"
        
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
        """Get the directory path for a specific video."""
        # Replace invalid characters in video title
        safe_title = "".join(c for c in video_title if c.isalnum() or c in (' ', '-', '_')).strip()
        return os.path.join(self.data_dir, safe_title)

    async def save_original_transcript(self, transcript: List[Dict], video_title: str) -> str:
        """Save the original transcript to a file."""
        video_dir = self._get_video_dir(video_title)
        os.makedirs(video_dir, exist_ok=True)
        
        filepath = os.path.join(video_dir, f"{video_title}_transcript_original.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(transcript, f, indent=2, ensure_ascii=False)
        
        return filepath

    async def save_analysis(self, analysis: str, video_title: str, analysis_type: str, model: str = "mistral") -> str:
        """Save the analysis result to a file."""
        video_dir = self._get_video_dir(video_title)
        os.makedirs(video_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_prefix = f"{model}_"  # Always include model name in filename
        filename = f"{video_title}_{model_prefix}analysis_{analysis_type}_{timestamp}.json"
        filepath = os.path.join(video_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'title': video_title,
                'type': analysis_type,
                'model': model,
                'timestamp': timestamp,
                'content': analysis
            }, f, indent=2, ensure_ascii=False)
        
        return filepath

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
                "Summarize the video's technical content with a focus on:\n"
                "- Core concepts discussed.\n"
                "- Methodologies used or proposed.\n"
                "- Key takeaways that someone should remember.\n"
                "- Any notable insights that differentiate this content from common knowledge.\n\n"
                "Organize the summary into clear sections with concise bullet points for readability."
            ),
            "code_snippets": (
                "Extract all code snippets and provide:\n"
                "1. The exact code (if present in transcription).\n"
                "2. The purpose of the code in the given context.\n"
                "3. A step-by-step explanation of how it works.\n"
                "4. Any assumptions or dependencies needed to run it.\n"
                "5. Best practices or optimizations if relevant.\n\n"
                "Ensure explanations are precise and useful for implementation."
            ),
            "tools_and_resources": (
                "Extract and list all tools, libraries, frameworks, and resources mentioned.\n"
                "For each, include:\n"
                "- The name and a brief description of what it does.\n"
                "- Why it was mentioned (e.g., problem it solves, improvement it provides).\n"
                "- Any links given or well-known official resources.\n"
                "- Common alternatives (if applicable)."
            ),
            "key_workflows": (
                "Identify and outline all workflows, methodologies, and structured processes mentioned.\n"
                "For each:\n"
                "1. Provide a step-by-step breakdown in bullet points.\n"
                "2. Highlight any dependencies or prerequisites.\n"
                "3. Note any best practices or warnings.\n"
                "4. If the workflow improves efficiency or solves a specific problem, explain how."
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
                "You are an expert in technical content analysis. Your goal is to extract only the most valuable insights from the video.\n\n"
                "Summarize the content with a focus on:\n"
                "- The most critical concepts, methodologies, and techniques.\n"
                "- Unique or advanced insights beyond basic knowledge.\n"
                "- Real-world applications mentioned.\n"
                "- Key takeaways that should be remembered.\n\n"
                "Format the output into sections with bullet points for clarity.\n"
                "Avoid unnecessary details—focus on actionable knowledge."
            ),
            "code_snippets": (
                "You are a code extraction and explanation expert.\n\n"
                "Extract all code snippets and provide:\n"
                "1. The full code (if available in the transcript).\n"
                "2. What the code does and why it matters.\n"
                "3. A step-by-step breakdown of its logic.\n"
                "4. Best practices, potential pitfalls, and optimizations.\n"
                "5. Any related concepts or frameworks necessary for implementation.\n\n"
                "Your goal is to provide quick, actionable insights for a developer reviewing this information."
            ),
            "tools_and_resources": (
                "You are a technical resource curator.\n\n"
                "Extract all tools, libraries, and frameworks mentioned in the video and provide:\n"
                "- Name\n"
                "- What it does\n"
                "- Why it was mentioned (e.g., advantage over alternatives, use case)\n"
                "- Links if provided\n"
                "- Related or competing technologies\n\n"
                "Focus on practical utility—omit anything that is just a passing mention without real significance."
            ),
            "key_workflows": (
                "You are a workflow and methodology analyst.\n\n"
                "Identify and break down all key workflows and structured processes discussed in the video.\n"
                "For each:\n"
                "1. Provide a step-by-step breakdown with clear formatting.\n"
                "2. Identify critical decisions, optimizations, or best practices.\n"
                "3. Highlight common mistakes, pitfalls, or challenges.\n"
                "4. If a workflow improves efficiency or solves a specific problem, explain how and why.\n\n"
                "Your goal is to help someone understand and apply the process immediately."
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

    async def chat_with_mistral(self, messages):
        """Chat with Mistral model using provided messages."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.ollama_base_url}/api/chat",
                    json={
                        "model": "mistral",
                        "messages": messages,
                        "stream": False
                    }
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        if 'message' in result:
                            return result['message']['content']
                        else:
                            logger.error(f"Unexpected API response structure: {result}")
                            raise Exception("Invalid response format from Ollama API")
                    else:
                        error_text = await response.text()
                        raise Exception(f"Mistral API error: {error_text}")
        except Exception as e:
            logger.error(f"Error in Mistral chat: {str(e)}", exc_info=True)
            raise 