"""Integrate Mistral OCR"""

import logging
from typing import Dict, List, Any, BinaryIO
from pathlib import Path

logger = logging.getLogger(__name__)


class OCRService:
    def __init__(self, api_key: str):
        self.api_key = api_key
        logger.info("OCR service initilized")


    def process_image(self, image_data: BinaryIO) -> Dict[str, Any]:
        """Process the image the ocr api"""
