import requests
from typing import Dict, Any
import json
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RedditExtractor:
    """Class for extracting content from Reddit posts."""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def extract_post_id(self, url: str) -> str:
        """Extract post ID from Reddit URL."""
        try:
            # Remove trailing slash if present
            if url.endswith('/'):
                url = url[:-1]
            # Extract ID from URL
            return url.split('/comments/')[1].split('/')[0]
        except Exception as e:
            logger.error(f"Failed to extract post ID from URL {url}: {str(e)}")
            raise ValueError(f"Invalid Reddit URL format: {url}")
    
    def extract_post(self, url: str) -> Dict[str, Any]:
        """Extract data from a Reddit post."""
        if not url.endswith('/'):
            url += '/'
        
        try:
            # Get JSON version of the post
            json_url = url + '.json'
            response = requests.get(json_url, headers=self.headers)
            response.raise_for_status()
            
            json_data = response.json()
            post_data = json_data[0]['data']['children'][0]['data']
            
            # Extract relevant information
            extracted_data = {
                'title': post_data.get('title'),
                'author': post_data.get('author'),
                'subreddit': post_data.get('subreddit'),
                'score': post_data.get('score'),
                'upvote_ratio': post_data.get('upvote_ratio'),
                'created_utc': datetime.fromtimestamp(post_data.get('created_utc')).isoformat(),
                'content': post_data.get('selftext'),
                'url': url,
                'is_self': post_data.get('is_self'),
                'permalink': f"https://reddit.com{post_data.get('permalink')}",
                'num_comments': post_data.get('num_comments'),
                'post_id': self.extract_post_id(url)
            }
            
            logger.info(f"Successfully extracted Reddit post: {extracted_data['title']}")
            return extracted_data
            
        except requests.RequestException as e:
            logger.error(f"Failed to fetch Reddit post: {str(e)}")
            raise
        except (KeyError, IndexError) as e:
            logger.error(f"Failed to parse Reddit data: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise 