"""Main application entry point."""
import logging
from app.config import get_config

logger = logging.getLogger(__name__)

def initialize_application():
    """Initialize the app components."""
    config = get_config()
    logger.info("Application initialized successfully!")
    return config

def main():
    try:
        initialize_application()
        logger.info("Receipt Analyzer application started")
    except Exception as e:
        logger.exception(f"Failed to start application: {e}")
        raise

if __name__ == "__main__":
    main()
