from typing import List, Dict, Optional, Any
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import json
import os
from datetime import datetime
from pathlib import Path
import logging
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoSegment(BaseModel):
    """Represents a segment of video content for the knowledge base."""
    content: str
    timestamp: str
    video_title: str
    video_id: str
    segment_type: str = Field(default="transcript")  # transcript, analysis, summary, etc.
    metadata: Dict[str, Any] = Field(default_factory=dict)
    source_file: str
    creation_date: str = Field(default_factory=lambda: datetime.now().isoformat())

class KnowledgeBase:
    def __init__(self, persist_directory: str = "data/knowledge_base"):
        """Initialize the knowledge base with Chroma."""
        self.persist_directory = persist_directory
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize Chroma client with persistence
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Use sentence-transformers for embeddings
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Initialize collections
        self.transcript_collection = self.client.get_or_create_collection(
            name="transcripts",
            embedding_function=self.embedding_function,
            metadata={"description": "Video transcript segments"}
        )
        
        self.analysis_collection = self.client.get_or_create_collection(
            name="analyses",
            embedding_function=self.embedding_function,
            metadata={"description": "Video analyses and summaries"}
        )
        
        # Initialize text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def process_transcript(self, transcript: List[Dict], video_title: str, video_id: str, source_file: str) -> None:
        """Process and store transcript segments in the knowledge base."""
        try:
            # Combine nearby segments for better context
            chunks = self.text_splitter.create_documents([
                seg['text'] for seg in transcript
            ])
            
            # Prepare segments for storage
            ids = []
            texts = []
            metadatas = []
            
            for i, chunk in enumerate(chunks):
                # Find corresponding timestamp for chunk
                timestamp = self._find_timestamp_for_chunk(chunk.page_content, transcript)
                
                segment = VideoSegment(
                    content=chunk.page_content,
                    timestamp=timestamp,
                    video_title=video_title,
                    video_id=video_id,
                    segment_type="transcript",
                    source_file=source_file
                )
                
                ids.append(f"{video_id}_transcript_{i}")
                texts.append(segment.content)
                metadatas.append(segment.dict(exclude={'content'}))
            
            # Store in Chroma
            self.transcript_collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas
            )
            
            logger.info(f"Successfully processed transcript for video: {video_title}")
            
        except Exception as e:
            logger.error(f"Error processing transcript: {str(e)}")
            raise
    
    def process_analysis(self, analysis: Dict, video_title: str, video_id: str, 
                        analysis_type: str, source_file: str) -> None:
        """Process and store analysis results in the knowledge base."""
        try:
            # Split analysis into chunks
            chunks = self.text_splitter.create_documents([analysis['content']])
            
            # Prepare segments for storage
            ids = []
            texts = []
            metadatas = []
            
            for i, chunk in enumerate(chunks):
                segment = VideoSegment(
                    content=chunk.page_content,
                    timestamp="",  # Analyses might not have specific timestamps
                    video_title=video_title,
                    video_id=video_id,
                    segment_type=f"analysis_{analysis_type}",
                    source_file=source_file,
                    metadata={"analysis_type": analysis_type}
                )
                
                ids.append(f"{video_id}_analysis_{analysis_type}_{i}")
                texts.append(segment.content)
                metadatas.append(segment.dict(exclude={'content'}))
            
            # Store in Chroma
            self.analysis_collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas
            )
            
            logger.info(f"Successfully processed {analysis_type} analysis for video: {video_title}")
            
        except Exception as e:
            logger.error(f"Error processing analysis: {str(e)}")
            raise
    
    def semantic_search(self, query: str, limit: int = 5, 
                       collection_name: str = "transcripts") -> List[Dict]:
        """Perform semantic search across the knowledge base."""
        try:
            collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            
            results = collection.query(
                query_texts=[query],
                n_results=limit,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'relevance_score': 1 - results['distances'][0][i]  # Convert distance to similarity score
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error performing semantic search: {str(e)}")
            raise
    
    def _find_timestamp_for_chunk(self, chunk_text: str, transcript: List[Dict]) -> str:
        """Find the appropriate timestamp for a chunk of text."""
        # Simple implementation - find first matching segment
        # Could be improved with more sophisticated matching
        chunk_words = set(chunk_text.lower().split())
        
        for segment in transcript:
            segment_words = set(segment['text'].lower().split())
            if len(chunk_words.intersection(segment_words)) > len(chunk_words) * 0.5:
                minutes = int(segment['start'] // 60)
                seconds = int(segment['start'] % 60)
                return f"[{minutes:02d}:{seconds:02d}]"
        
        return ""  # Return empty string if no matching timestamp found 