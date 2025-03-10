from typing import List, Dict, Optional, Any, Union
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
from enum import Enum
import re
from typing import List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_text(text: str, is_code: bool = False) -> str:
    """Clean text before embedding."""
    if is_code:
        # For code, preserve case and indentation but remove redundant whitespace
        return '\n'.join(line.rstrip() for line in text.split('\n'))
    
    # For regular text:
    # 1. Convert to lowercase
    text = text.lower()
    # 2. Remove redundant whitespace
    text = ' '.join(text.split())
    # 3. Remove redundant punctuation
    text = re.sub(r'([.,!?])\1+', r'\1', text)
    # 4. Normalize quotes and dashes
    text = text.replace('"', '"').replace('"', '"').replace('--', '—')
    return text

def preprocess_chunks(chunks: List[str], is_code: bool = False) -> List[str]:
    """Process a list of text chunks."""
    processed_chunks = []
    for chunk in chunks:
        # Skip empty chunks
        if not chunk.strip():
            continue
        # Clean the text
        cleaned = clean_text(chunk, is_code)
        # Skip if chunk became empty after cleaning
        if cleaned:
            processed_chunks.append(cleaned)
    return processed_chunks

class ContentType(str, Enum):
    """Types of content that can be stored in the knowledge base."""
    TRANSCRIPT = "transcript"
    TECHNICAL_SUMMARY = "technical_summary"
    CODE_SNIPPET = "code_snippet"
    TOOL_REFERENCE = "tool_reference"
    CONCEPT_EXPLANATION = "concept_explanation"
    IMPLEMENTATION_EXAMPLE = "implementation_example"
    BEST_PRACTICE = "best_practice"
    VERSION_INFO = "version_info"
    TUTORIAL = "tutorial"

class ContentMetadata(BaseModel):
    """Metadata for any piece of content in the knowledge base."""
    content_type: ContentType
    source_type: str  # video, documentation, article, etc.
    source_id: str  # video_id, doc_id, etc.
    source_title: str
    source_url: Optional[str] = None
    timestamp: Optional[str] = None  # For video content
    creation_date: str = Field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = Field(default_factory=lambda: datetime.now().isoformat())
    version_info: Optional[Dict[str, str]] = None  # For tool/library versions
    tags: List[str] = Field(default_factory=list)
    difficulty_level: Optional[str] = None
    prerequisites: List[str] = Field(default_factory=list)
    related_concepts: List[str] = Field(default_factory=list)
    additional_metadata: Dict[str, Any] = Field(default_factory=dict)

class KnowledgeContent(BaseModel):
    """Represents a piece of content in the knowledge base."""
    id: str
    content: str
    metadata: ContentMetadata

class RAGModel:
    def __init__(self, persist_directory: str = "data/knowledge_base"):
        """Initialize the RAG model with Chroma."""
        self.persist_directory = persist_directory
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize Chroma client with persistence
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Use sentence-transformers for embeddings
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-mpnet-base-v2"
        )
        
        # Initialize collections for different types of content
        self.collections = {}
        for content_type in ContentType:
            self.collections[content_type] = self.client.get_or_create_collection(
                name=f"{content_type.value}_collection",
                embedding_function=self.embedding_function,
                metadata={"content_type": content_type.value}
            )
        
        # Initialize text splitters with token-based splitting
        self.splitters = {
            ContentType.TRANSCRIPT: RecursiveCharacterTextSplitter(
                chunk_size=384,  # MPNet optimal token size
                chunk_overlap=50,
                separators=["\n\n", "\n", ".", "!", "?", " ", ""]
            ),
            ContentType.CODE_SNIPPET: RecursiveCharacterTextSplitter(
                chunk_size=256,  # Smaller for code to maintain context
                chunk_overlap=30,
                separators=["\n\n", "\n", ";", "}", "{"]
            ),
            ContentType.CONCEPT_EXPLANATION: RecursiveCharacterTextSplitter(
                chunk_size=384,
                chunk_overlap=100,
                separators=["\n\n", "\n", ".", "!", "?", " "]
            )
        }
        
        # Default splitter for other content types
        self.default_splitter = RecursiveCharacterTextSplitter(
            chunk_size=384,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", "!", "?", " ", ""]
        )
    
    def add_content(self, content: Union[str, List[str]], metadata: ContentMetadata) -> List[str]:
        """Add content to the knowledge base."""
        try:
            # Get appropriate splitter
            splitter = self.splitters.get(metadata.content_type, self.default_splitter)
            
            # Convert content to list if it's a single string
            content_list = content if isinstance(content, list) else [content]
            
            # Create chunks
            chunks = []
            for text in content_list:
                chunks.extend(splitter.create_documents([text]))
            
            # Process chunks based on content type
            is_code = metadata.content_type in [ContentType.CODE_SNIPPET, ContentType.IMPLEMENTATION_EXAMPLE]
            processed_chunks = preprocess_chunks([chunk.page_content for chunk in chunks], is_code)
            
            # Prepare data for Chroma
            ids = []
            texts = []
            metadatas = []
            
            for i, chunk_text in enumerate(processed_chunks):
                content_id = f"{metadata.source_id}_{metadata.content_type}_{i}"
                ids.append(content_id)
                texts.append(chunk_text)
                
                # Create metadata for each chunk
                chunk_metadata = metadata.dict(exclude_none=True)  # Exclude None values
                
                # Convert lists and dicts to strings in metadata
                for key, value in chunk_metadata.items():
                    if isinstance(value, (list, dict)):
                        chunk_metadata[key] = json.dumps(value)
                
                chunk_metadata['chunk_index'] = i
                chunk_metadata['total_chunks'] = len(processed_chunks)
                metadatas.append(chunk_metadata)
            
            # Store in appropriate collection
            collection = self.collections[metadata.content_type]
            collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas
            )
            
            logger.info(f"Successfully added {len(processed_chunks)} chunks of {metadata.content_type} content")
            return ids
            
        except Exception as e:
            logger.error(f"Error adding content: {str(e)}")
            raise
    
    def search(self, 
              query: str, 
              content_types: Optional[List[ContentType]] = None,
              filters: Optional[Dict[str, Any]] = None,
              limit: int = 5) -> List[Dict]:
        """
        Search across specified content types with optional filters.
        
        Args:
            query: Search query
            content_types: List of ContentType to search in, or None for all
            filters: Dictionary of metadata filters
            limit: Maximum number of results per collection
        """
        try:
            results = []
            search_types = content_types or list(ContentType)
            
            for content_type in search_types:
                collection = self.collections[content_type]
                
                # Perform search in collection
                collection_results = collection.query(
                    query_texts=[query],
                    n_results=limit,
                    where=filters,  # Chroma will handle metadata filtering
                    include=["documents", "metadatas", "distances"]
                )
                
                # Format and add results
                for i in range(len(collection_results['documents'][0])):
                    results.append({
                        'content': collection_results['documents'][0][i],
                        'metadata': collection_results['metadatas'][0][i],
                        'relevance_score': 1 - collection_results['distances'][0][i],
                        'content_type': content_type.value
                    })
            
            # Sort all results by relevance score
            results.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            # Return top results across all collections
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Error performing search: {str(e)}")
            raise
    
    def get_related_content(self, content_id: str, limit: int = 5) -> List[Dict]:
        """Find content related to a specific piece of content."""
        try:
            # First, get the original content
            for collection in self.collections.values():
                try:
                    original = collection.get(
                        ids=[content_id],
                        include=["documents", "metadatas"]
                    )
                    if original['ids']:
                        break
                except:
                    continue
            
            if not original['ids']:
                raise ValueError(f"Content with id {content_id} not found")
            
            # Use the content as a search query
            return self.search(
                query=original['documents'][0],
                filters={"id": {"$ne": content_id}},  # Exclude the original content
                limit=limit
            )
            
        except Exception as e:
            logger.error(f"Error getting related content: {str(e)}")
            raise
    
    def update_content(self, content_id: str, new_content: str, 
                      metadata_updates: Optional[Dict[str, Any]] = None) -> None:
        """Update existing content and its metadata."""
        try:
            # Find the content in collections
            for content_type, collection in self.collections.items():
                try:
                    existing = collection.get(
                        ids=[content_id],
                        include=["metadatas"]
                    )
                    if existing['ids']:
                        # Get existing metadata
                        metadata = existing['metadatas'][0]
                        
                        # Update metadata if provided
                        if metadata_updates:
                            metadata.update(metadata_updates)
                            metadata['last_updated'] = datetime.now().isoformat()
                        
                        # Delete existing content
                        collection.delete(ids=[content_id])
                        
                        # Add updated content
                        self.add_content(
                            content=new_content,
                            metadata=ContentMetadata(**metadata)
                        )
                        
                        logger.info(f"Successfully updated content {content_id}")
                        return
                except:
                    continue
            
            raise ValueError(f"Content with id {content_id} not found")
            
        except Exception as e:
            logger.error(f"Error updating content: {str(e)}")
            raise
    
    def delete_content(self, content_id: str) -> None:
        """Delete content from the knowledge base."""
        try:
            deleted = False
            for collection in self.collections.values():
                try:
                    collection.delete(ids=[content_id])
                    deleted = True
                except:
                    continue
            
            if not deleted:
                raise ValueError(f"Content with id {content_id} not found")
            
            logger.info(f"Successfully deleted content {content_id}")
            
        except Exception as e:
            logger.error(f"Error deleting content: {str(e)}")
            raise 