import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from scripts.rag_model import RAGModel, ContentMetadata, ContentType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MigrationAdapter:
    def __init__(self, data_directory: str = "data", rag_model: Optional[RAGModel] = None):
        """Initialize the migration adapter."""
        self.data_directory = Path(data_directory)
        self.rag_model = rag_model or RAGModel()
        
    def migrate_all_content(self) -> Tuple[int, int]:
        """
        Migrate all existing content to the new RAG model.
        Returns tuple of (success_count, error_count)
        """
        success_count = 0
        error_count = 0
        
        # Iterate through all video directories
        for video_dir in self.data_directory.iterdir():
            if not video_dir.is_dir():
                continue
                
            try:
                # Get video metadata
                metadata_file = next(video_dir.glob("*__metadata__*.json"), None)
                if not metadata_file:
                    logger.warning(f"No metadata found for {video_dir.name}")
                    continue
                    
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    video_metadata = json.load(f)
                
                # Migrate transcript
                transcript_success = self._migrate_transcript(video_dir, video_metadata)
                success_count += 1 if transcript_success else 0
                error_count += 0 if transcript_success else 1
                
                # Migrate analyses
                analysis_success, analysis_errors = self._migrate_analyses(video_dir, video_metadata)
                success_count += analysis_success
                error_count += analysis_errors
                
                logger.info(f"Successfully processed directory: {video_dir.name}")
                
            except Exception as e:
                logger.error(f"Error processing directory {video_dir.name}: {str(e)}")
                error_count += 1
        
        return success_count, error_count
    
    def _migrate_transcript(self, video_dir: Path, video_metadata: Dict) -> bool:
        """Migrate a video transcript to the new RAG model."""
        try:
            # Find most recent transcript file
            transcript_file = next(video_dir.glob("*__transcript__*.json"), None)
            if not transcript_file:
                return False
            
            with open(transcript_file, 'r', encoding='utf-8') as f:
                transcript = json.load(f)
            
            # Create metadata for transcript
            metadata = ContentMetadata(
                content_type=ContentType.TRANSCRIPT,
                source_type="video",
                source_id=video_metadata.get('video_id', video_dir.name),
                source_title=video_metadata.get('title', video_dir.name),
                source_url=video_metadata.get('video_url', ''),
                creation_date=datetime.fromtimestamp(transcript_file.stat().st_mtime).isoformat(),
                tags=self._extract_tags(video_metadata),
                additional_metadata={
                    'channel_name': video_metadata.get('channel_name', 'unknown'),
                    'publish_date': video_metadata.get('publish_date', ''),
                    'view_count': video_metadata.get('view_count', 0)
                }
            )
            
            # Extract text content from transcript
            content = [seg['text'] for seg in transcript]
            
            # Add to RAG model
            self.rag_model.add_content(content, metadata)
            logger.info(f"Successfully migrated transcript for {video_metadata.get('title')}")
            return True
            
        except Exception as e:
            logger.error(f"Error migrating transcript: {str(e)}")
            return False
    
    def _migrate_analyses(self, video_dir: Path, video_metadata: Dict) -> Tuple[int, int]:
        """Migrate video analyses to the new RAG model."""
        success_count = 0
        error_count = 0
        
        try:
            # Map old analysis types to new content types
            analysis_type_mapping = {
                'technical_summary': ContentType.TECHNICAL_SUMMARY,
                'code_snippets': ContentType.CODE_SNIPPET,
                'tools_and_resources': ContentType.TOOL_REFERENCE,
                'key_workflows': ContentType.IMPLEMENTATION_EXAMPLE
            }
            
            # Process each analysis file
            for analysis_file in video_dir.glob("*__analysis__*.json"):
                try:
                    # Parse analysis type from filename
                    parts = analysis_file.name.split('__')
                    if len(parts) < 4:
                        continue
                        
                    old_analysis_type = parts[2]
                    model_used = parts[3]
                    
                    # Map to new content type
                    content_type = analysis_type_mapping.get(old_analysis_type)
                    if not content_type:
                        continue
                    
                    # Load analysis content
                    with open(analysis_file, 'r', encoding='utf-8') as f:
                        analysis = json.load(f)
                    
                    # Create metadata
                    metadata = ContentMetadata(
                        content_type=content_type,
                        source_type="video",
                        source_id=video_metadata.get('video_id', video_dir.name),
                        source_title=video_metadata.get('title', video_dir.name),
                        source_url=video_metadata.get('video_url', ''),
                        creation_date=datetime.fromtimestamp(analysis_file.stat().st_mtime).isoformat(),
                        tags=self._extract_tags(video_metadata),
                        additional_metadata={
                            'analysis_model': model_used,
                            'original_type': old_analysis_type,
                            'channel_name': video_metadata.get('channel_name', 'unknown')
                        }
                    )
                    
                    # Add to RAG model
                    self.rag_model.add_content(analysis['content'], metadata)
                    success_count += 1
                    logger.info(f"Successfully migrated {old_analysis_type} analysis for {video_metadata.get('title')}")
                    
                except Exception as e:
                    logger.error(f"Error migrating analysis file {analysis_file.name}: {str(e)}")
                    error_count += 1
            
        except Exception as e:
            logger.error(f"Error in _migrate_analyses: {str(e)}")
            error_count += 1
        
        return success_count, error_count
    
    def _extract_tags(self, video_metadata: Dict) -> List[str]:
        """Extract relevant tags from video metadata."""
        tags = []
        
        # Add channel name as tag
        if channel := video_metadata.get('channel_name'):
            tags.append(f"channel:{channel}")
        
        # Extract keywords from title and description
        title = video_metadata.get('title', '').lower()
        description = video_metadata.get('description', '').lower()
        
        # Keywords to look for
        keywords = ['llm', 'ai', 'ml', 'gpt', 'transformer', 'neural', 'deep learning',
                   'python', 'javascript', 'typescript', 'react', 'node', 'api',
                   'langchain', 'openai', 'anthropic', 'claude', 'mistral', 'gemini']
        
        # Add found keywords as tags
        for keyword in keywords:
            if keyword in title or keyword in description:
                tags.append(keyword)
        
        return list(set(tags))  # Remove duplicates 