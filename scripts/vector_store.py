import os
import json
import logging
import hashlib
from typing import List, Dict, Optional, Any, Tuple
import nltk
from nltk.tokenize import sent_tokenize
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VectorStore:
    """Vector store for transcript segments with enhanced context and metadata."""
    
    def __init__(self, persist_directory: str = "chroma_db"):
        """Initialize the vector store with the specified persistence directory."""
        try:
            # Download NLTK data if needed
            nltk.download('punkt', quiet=True)
            
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(path=persist_directory)
            
            # Use sentence-transformers model for embeddings
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="BAAI/bge-m3"
            )
            
            # Create or get the collection
            try:
                self.collection = self.client.get_collection(
                    name="transcript_segments",
                    embedding_function=self.embedding_function
                )
                logger.info(f"Using existing collection with {self.collection.count()} items")
            except Exception:
                self.collection = self.client.create_collection(
                    name="transcript_segments",
                    embedding_function=self.embedding_function
                )
                logger.info("Created new collection")
                
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            raise
    
    def _get_semantic_segments(self, text: str, min_size: int = 100, max_size: int = 500) -> List[str]:
        """Split text into semantic segments based on sentence boundaries."""
        try:
            # Split text into sentences
            sentences = sent_tokenize(text)
            
            segments = []
            current_segment = ""
            
            for sentence in sentences:
                # If adding this sentence would exceed max_size, start a new segment
                if len(current_segment) + len(sentence) > max_size and len(current_segment) >= min_size:
                    segments.append(current_segment.strip())
                    current_segment = sentence
                else:
                    current_segment += " " + sentence if current_segment else sentence
            
            # Add the last segment if it's not empty
            if current_segment:
                segments.append(current_segment.strip())
                
            return segments
            
        except Exception as e:
            logger.error(f"Error creating semantic segments: {str(e)}")
            return [text]  # Return original text as a single segment if segmentation fails
    
    def _get_window_size(self, content_type: str = "general") -> int:
        """Get the context window size based on content type."""
        # Define window sizes for different content types
        window_sizes = {
            "transcript": 2,  # Regular transcript segments
            "technical_summary": 0,  # Technical summaries don't need context
            "full_context": 0,  # Full context already has all information
            "code_snippets": 1,  # Code snippets benefit from some context
            "tools_and_resources": 0,  # Tools and resources are standalone
            "key_workflows": 1,  # Key workflows benefit from some context
            "general": 2  # Default window size
        }
        
        return window_sizes.get(content_type, window_sizes["general"])
    
    def _combine_segments(self, segments: List[Dict[str, Any]], content_type: str = "general") -> List[Dict[str, Any]]:
        """Combine segments with surrounding context based on content type."""
        try:
            # If no segments or only one segment, return as is
            if not segments or len(segments) <= 1:
                return segments
            
            # Get window size based on content type
            window_size = self._get_window_size(content_type)
            
            # If window size is 0, return segments as is
            if window_size == 0:
                return segments
            
            # Create enriched segments with context
            enriched_segments = []
            
            for i, segment in enumerate(segments):
                # Create a copy of the segment
                enriched_segment = segment.copy()
                
                # Add context from surrounding segments
                context_before = []
                context_after = []
                
                # Get context before current segment
                for j in range(max(0, i - window_size), i):
                    context_before.append(segments[j]["text"])
                
                # Get context after current segment
                for j in range(i + 1, min(len(segments), i + window_size + 1)):
                    context_after.append(segments[j]["text"])
                
                # Add context to segment
                if context_before:
                    enriched_segment["context_before"] = " ".join(context_before)
                if context_after:
                    enriched_segment["context_after"] = " ".join(context_after)
                
                enriched_segments.append(enriched_segment)
            
            return enriched_segments
            
        except Exception as e:
            logger.error(f"Error combining segments with context: {str(e)}")
            return segments  # Return original segments if combination fails
    
    def _enrich_metadata(self, video_id: str) -> Dict[str, Any]:
        """Enrich metadata with video information."""
        try:
            # Get video metadata from data directory
            video_dir = os.path.join("data", video_id)
            metadata_file = os.path.join(video_dir, "metadata.json")
            
            if os.path.exists(metadata_file):
                with open(metadata_file, "r", encoding="utf-8") as f:
                    video_metadata = json.load(f)
                
                # Extract relevant metadata fields
                return {
                    "title": video_metadata.get("title", ""),
                    "author": video_metadata.get("author", ""),
                    "upload_date": video_metadata.get("upload_date", ""),
                    "description": video_metadata.get("description", "")
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Error enriching metadata for video {video_id}: {str(e)}")
            return {}
    
    def _detect_content_type(self, text: str) -> str:
        """Detect content type based on text patterns."""
        text_lower = text.lower()
        
        # Check for code patterns
        code_indicators = ["def ", "class ", "import ", "function", "return ", "```python", "```javascript"]
        if any(indicator in text_lower for indicator in code_indicators):
            return "code_snippets"
        
        # Check for technical summary patterns
        technical_indicators = ["technical summary", "overview", "architecture", "implementation", "algorithm"]
        if any(indicator in text_lower for indicator in technical_indicators):
            return "technical_summary"
        
        # Check for tools and resources patterns
        tools_indicators = ["tool", "library", "framework", "resource", "package", "dependency"]
        if any(indicator in text_lower for indicator in tools_indicators):
            return "tools_and_resources"
        
        # Default to transcript
        return "transcript"
    
    def _expand_query(self, query: str) -> List[str]:
        """Expand a query into multiple variations for better retrieval."""
        variations = [query]  # Start with the original query
        
        # Add a "what is" variation for concept queries
        if not query.lower().startswith("what is") and not query.lower().startswith("how to"):
            variations.append(f"What is {query}?")
        
        # Add a "how to" variation for procedural queries
        if not query.lower().startswith("how to") and not query.lower().startswith("what is"):
            variations.append(f"How to {query}?")
        
        return variations
    
    def add_transcript(self, video_id: str, transcript_segments: List[Dict[str, Any]], batch_size: int = 100) -> None:
        """Add a transcript to the vector store with enhanced context and metadata."""
        try:
            # Enrich metadata with video information
            video_metadata = self._enrich_metadata(video_id)
            
            # Get total segments and log info
            total_segments = len(transcript_segments)
            logger.info(f"Processing {total_segments} segments for video {video_id}")
            
            # Process in batches
            for batch_start in range(0, total_segments, batch_size):
                batch_end = min(batch_start + batch_size, total_segments)
                batch = transcript_segments[batch_start:batch_end]
                
                # Prepare documents and metadata for this batch
                documents = []
                metadatas = []
                ids = []
                
                # Process each segment in the batch
                for idx, segment in enumerate(batch):
                    # Calculate global segment index
                    segment_index = batch_start + idx
                    
                    # Prepare metadata
                    metadata = {
                        "video_id": video_id,
                        "start": segment["start"],
                        "duration": segment["duration"],
                        "content_type": "transcript",
                        "segment_index": segment_index,
                        "total_segments": total_segments
                    }
                    
                    # Add video metadata
                    metadata.update(video_metadata)
                    
                    # Generate unique ID using global segment index
                    segment_id = f"{video_id}_{segment_index}"
                    
                    # Add to batch lists
                    documents.append(segment["text"])
                    metadatas.append(metadata)
                    ids.append(segment_id)
                
                # Add the batch to the collection
                try:
                    self.collection.add(
                        documents=documents,
                        metadatas=metadatas,
                        ids=ids
                    )
                    logger.info(f"Added batch for {video_id} (segments {batch_start}-{batch_end-1})")
                except Exception as e:
                    logger.error(f"Error adding batch for video {video_id}: {str(e)}")
                    raise
            
            logger.info(f"Successfully added all {total_segments} segments for {video_id}")
            
        except Exception as e:
            logger.error(f"Error processing video {video_id}: {str(e)}")
            raise
    
    def search(self, query: str, n_results: int = 5, filter_metadata: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Search for transcript segments matching the query."""
        try:
            # Expand query for better retrieval
            expanded_queries = self._expand_query(query)
            
            # Search with expanded queries
            results = self.collection.query(
                query_texts=expanded_queries,
                n_results=n_results,
                where=filter_metadata
            )
            
            # Process results
            processed_results = []
            
            for i, (doc_id, document, metadata, distance) in enumerate(zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )):
                # Calculate relevance score (1 - distance, normalized to [0, 1])
                relevance_score = 1 - min(distance, 1.0)
                
                # Create result object
                result = {
                    "id": doc_id,
                    "text": document,
                    "metadata": metadata,
                    "relevance_score": round(relevance_score, 3)
                }
                
                processed_results.append(result)
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Error searching for query '{query}': {str(e)}")
            return []
    
    def get_video_segments(self, video_id: str) -> List[Dict[str, Any]]:
        """Get all segments for a specific video."""
        try:
            # Query for all segments with the given video_id
            results = self.collection.query(
                query_texts=[""],  # Empty query to match all documents
                where={"video_id": video_id},
                n_results=10000  # Set a high limit to get all segments
            )
            
            # Process results
            segments = []
            
            for i, (doc_id, document, metadata) in enumerate(zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0]
            )):
                # Create segment object
                segment = {
                    "id": doc_id,
                    "text": document,
                    "start": metadata.get("start", 0),
                    "duration": metadata.get("duration", 0),
                    "segment_index": metadata.get("segment_index", i),
                    "metadata": metadata
                }
                
                segments.append(segment)
            
            # Sort segments by start time
            segments.sort(key=lambda x: x["start"])
            
            return segments
            
        except Exception as e:
            logger.error(f"Error getting segments for video {video_id}: {str(e)}")
            return []
    
    def delete_video(self, video_id: str) -> None:
        """Delete all segments for a specific video."""
        try:
            # Delete all documents with the given video_id
            self.collection.delete(
                where={"video_id": video_id}
            )
            logger.info(f"Deleted all segments for video {video_id}")
            
        except Exception as e:
            logger.error(f"Error deleting segments for video {video_id}: {str(e)}")
            raise
    
    def clear(self) -> None:
        """Clear all data from the vector store."""
        try:
            # Delete the collection
            self.client.delete_collection(name="transcript_segments")
            
            # Recreate the collection
            self.collection = self.client.create_collection(
                name="transcript_segments",
                embedding_function=self.embedding_function
            )
            
            logger.info("Cleared all data from vector store")
            
        except Exception as e:
            logger.error(f"Error clearing vector store: {str(e)}")
            raise 