import logging
from pathlib import Path
from scripts.migration_adapter import MigrationAdapter
from scripts.rag_model import RAGModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Run the migration from JSON files to RAG database."""
    try:
        # Initialize RAG model and migration adapter
        rag_model = RAGModel()
        adapter = MigrationAdapter(rag_model=rag_model)
        
        # Run migration
        logger.info("Starting migration...")
        success_count, error_count = adapter.migrate_all_content()
        
        # Log results
        logger.info(f"Migration completed:")
        logger.info(f"Successfully migrated {success_count} items")
        logger.info(f"Failed to migrate {error_count} items")
        
        if error_count > 0:
            logger.warning("Some items failed to migrate. Check the logs for details.")
        
    except Exception as e:
        logger.error(f"Migration failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 