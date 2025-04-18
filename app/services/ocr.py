"""OCR service for processing receipt images using Mistral OCR API."""
import logging
import io
import base64
from typing import Dict, Any, Optional, BinaryIO

import requests
from PIL import Image
from app.config import get_config

logger = logging.getLogger(__name__)

class OCRService:
    """Service for OCR processing using Mistral OCR API."""

    def __init__(self, api_key: Optional[str] = None):
        config = get_config()
        self.api_key = api_key
        self.api_url = "https://api.mistral.ai/v1/ocr"
        logger.info("OCR Service initialized")

    def _encode_image(self, image_path: str) -> str:
        """Encode an image file to base64"""

        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def _preprocess_image(self, image_data: BinaryIO) -> BinaryIO:
        """Preprocess the image"""
        # Open image with PIL
        image = Image.open(image_data)

        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Resize if too large (keeping aspect ratio)
        max_size = 2000  # Maximum width or height
        if image.width > max_size or image.height > max_size:
            ratio = min(max_size / image.width, max_size / image.height)
            new_size = (int(image.width * ratio), int(image.height * ratio))
            image = image.resize(new_size, Image.LANCZOS)

        # Save to bytes buffer
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=95)
        buffer.seek(0)

        return buffer


    def process_image(self, image_path: str) -> Dict[str, any]:
        """Process an image through the OCR API"""

        logger.info(f"Processing image with OCR: {image_path}")
        try:
            base64_image = self._encode_image(image_path)


            # Prepare the request payload
            payload = {
                "image": base64_image,
                "options": {
                    "text_extraction": True,
                    "structure_analysis": True
                }
            }

            # Set up headers
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            # Make the API request
            response = requests.post(
                self.api_url,
                json=payload,
                headers=headers
            )

            # Check for errors
            response.raise_for_status()

            # Return the JSON response
            result = response.json()
            logger.info("OCR processing completed successfully")
            return result

        except requests.RequestException as e:
            logger.error(f"OCR API request failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            raise

    def process_uploaded_file(self, uploaded_file: BinaryIO) -> Dict[str, Any]:
        """Process an uploaded file through the OCR API"""

        logger.info("Processing uploaded file with OCR")

        try:
            # Preprocess the image
            preprocessed_image = self._preprocess_image(uploaded_file)

            # Convert to base64
            base64_image = base64.b64encode(preprocessed_image.read()).decode("utf-8")

            # Prepare the request payload
            payload = {
                "image": base64_image,
                "options": {
                    "text_extraction": True,
                    "structure_analysis": True
                }
            }

            # Set up headers
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            # Make the API request
            response = requests.post(
                self.api_url,
                json=payload,
                headers=headers
            )

            # Check for errors
            response.raise_for_status()

            # Return the JSON response
            result = response.json()
            logger.info("OCR processing completed successfully")
            return result

        except requests.RequestException as e:
            logger.error(f"OCR API request failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Error processing uploaded file: {e}")
            raise

    def extract_receipt_data(self, ocr_result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract structured receipt data from OCR results"""

        logger.info("Extracting receipt data from OCR results")

        extracted_text = ocr_result.get("text", "")

        # return a placeholder with the raw text
        receipt_data = {
            "ocr_text": extracted_text,
            "vendor": "",
            "date": None,
            "items": [],
            "total": 0.0,
        }

        return receipt_data



