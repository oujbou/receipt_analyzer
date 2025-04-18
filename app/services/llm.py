"""Process the OCR results with Mistral LLM"""

import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime

#from mistralai.client import MistralClient
# from mistralai.models.chat_completion import ChatMessage
from mistralai import Mistral, UserMessage, SystemMessage


from app.config import get_config
from app.models.receipt import Receipt, ReceiptItem

logger = logging.getLogger(__name__)


class LLMService:
    """Service for LLM Operations using Mistral"""

    def __init__(self, api_key: Optional[str] = None, model: str = "mistral-large-latest"):
        config = get_config()
        self.api_key = api_key or config.mistral.api_key
        self.model = model
        self.client = Mistral(api_key=api_key)
        logger.info(f"LLM Service initialized with model: {model}")

    def _call_mistral(self, messages: List[Dict[str, str]], temperature: float = 0.1) -> str:
        """Call the Mistral API with chat messages"""

        try:
            response = self.client.chat.complete(
                model=self.model,
                messages=messages,
                temperature=temperature
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error calling Mistral API: {e}")
            raise

    def extract_receipt_data(self, ocr_text: str) -> Dict[str, any]:
        """Extract structured receipt data from ocr text"""
        logger.info("Extracting structured data from OCR text")

        # Prompt to extract details from the receipt
        messages = [
            {
                "role": "system",
                "content": """You are an expert receipt analyzer. Extract structured data from receipt text.
                        Extract the following information:
                        - Vendor name
                        - Date (in YYYY-MM-DD format)
                        - List of items with name, price, and quantity if available
                        - Subtotal (if present)
                        - Tax amount (if present)
                        - Total amount

                        Return the data as a JSON object with these keys: vendor, date, items, subtotal, tax, total.
                        For items, use an array of objects with keys: name, price, quantity.
                        If information is missing or unclear, use null for that field.
                        """
            },
            {
                "role": "user",
                "content": f"Extract data from this receipt text:\n\n{ocr_text}"
            }
        ]

        # Call the model
        response_text = self._call_mistral(messages)

        try:
            # extract JSON from the response (handle cases where LLM might add commentary)
            json_str = response_text
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_str = response_text.split("```")[1].split("```")[0].strip()

            receipt_data = json.loads(json_str)

            # Ensure all required fields are present
            if "vendor" not in receipt_data:
                receipt_data["vendor"] = "Unknown Vendor"

            if "date" not in receipt_data or not receipt_data["date"]:
                receipt_data["date"] = datetime.now().strftime("%Y-%m-%d")

            if "total" not in receipt_data or not receipt_data["total"]:
                # Infer total from items if possible
                if "items" in receipt_data and receipt_data["items"]:
                    total = sum(item.get("price", 0) * item.get("quantity", 1)
                                for item in receipt_data["items"])
                    receipt_data["total"] = total
                else:
                    receipt_data["total"] = 0.0

            # Add the original OCR text
            receipt_data["ocr_text"] = ocr_text

            logger.info("Successfully extracted structured receipt data")
            return receipt_data

        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            # Fallback to a basic structure
            return {
                "vendor": "Unknown Vendor",
                "date": datetime.now().strftime("%Y-%m-%d"),
                "items": [],
                "total": 0.0,
                "ocr_text": ocr_text
            }

    def classify_expenses(self, receipt_data: Dict[str, Any]) -> Dict[str, Any]:
        """Classify items in a receipt into expense categories"""
        logger.info("Classifying expenses")

        if "items" not in receipt_data or not receipt_data["items"]:
            logger.warning("No items to classify")
            return receipt_data

        item_names = [item.get("name", "") for item in receipt_data["items"]]

        messages = [
            {
                "role": "system",
                "content": """You are an expert expense classifier. Classify each item into one of these categories:
                        - Food & Dining
                        - Groceries
                        - Transportation
                        - Utilities
                        - Office Supplies
                        - Electronics
                        - Services
                        - Entertainment
                        - Travel
                        - Other

                        Return a JSON array with categories in the same order as the input items.
                        """
            },
            {
                "role": "user",
                "content": f"Classify these receipt items:\n\n{json.dumps(item_names)}"
            }
        ]

        try:
            response_text = self._call_mistral(messages)

            # Extract JSON from response
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_str = response_text.split("```")[1].split("```")[0].strip()
            else:
                json_str = response_text

            categories = json.loads(json_str)

            # Update the items with categories
            for i, category in enumerate(categories):
                if i < len(receipt_data["items"]):
                    receipt_data["items"][i]["category"] = category

            logger.info("Successfully classified expense items")
            return receipt_data

        except Exception as e:
            logger.error(f"Error classifying expenses: {e}")
            # Return original data if classification fails
            return receipt_data

    def validate_receipt(self, receipt_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate receipt data for consistency and errors"""
        logger.info("Validating receipt data")

        messages = [
            {
                "role": "system",
                "content": """You are an expert receipt validator. Check the receipt data for consistency and errors.
                        Specifically:
                        1. Verify that the total matches the sum of items (accounting for tax and subtotal if present)
                        2. Look for any unreasonable values (e.g., extremely high prices, negative values)
                        3. Suggest corrections for any issues found

                        Return a JSON object with these keys:
                        - valid: boolean indicating if the receipt is valid
                        - corrections: array of strings describing corrections
                        - corrected_data: the corrected receipt data (same format as input)
                        """
            },
            {
                "role": "user",
                "content": f"Validate this receipt data:\n\n{json.dumps(receipt_data, default=str)}"
            }
        ]
        try:
            # Call the model
            response_text = self._call_mistral(messages)

            # Extract JSON from response
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_str = response_text.split("```")[1].split("```")[0].strip()
            else:
                json_str = response_text

            validation_result = json.loads(json_str)

            # Use corrected data if available, or original otherwise
            validated_data = validation_result.get("corrected_data", receipt_data)

            # Add validation results
            validated_data["validation"] = {
                "valid": validation_result.get("valid", True),
                "corrections": validation_result.get("corrections", [])
            }

            logger.info(f"Receipt validation completed: Valid={validation_result.get('valid', True)}")
            return validated_data
        except Exception as e:
            logger.error(f"Error validating receipt: {e}")
            # Return original data with no corrections if validation fails
            receipt_data["validation"] = {
                "valid": True,
                "corrections": []
            }
            return receipt_data

    def create_receipt_object(self, receipt_data: Dict[str, Any]) -> Receipt:
        """Create a Receipt object from structured data"""
        logger.info("Creating Receipt object from structured data")

        try:
            # Convert string date to datetime
            date_str = receipt_data.get("date")
            if isinstance(date_str, str):
                date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
            else:
                date_obj = datetime.now().date()

            # Create ReceiptItem objects
            items = []
            for item_data in receipt_data.get("items", []):
                items.append(ReceiptItem(
                    name=item_data.get("name", "Unknown Item"),
                    price=float(item_data.get("price", 0)),
                    quantity=float(item_data.get("quantity", 1)),
                    category=item_data.get("category")
                ))

            # Create Receipt object
            receipt = Receipt(
                vendor=receipt_data.get("vendor", "Unknown Vendor"),
                date=date_obj,
                items=items,
                subtotal=receipt_data.get("subtotal"),
                tax=receipt_data.get("tax"),
                total=float(receipt_data.get("total", 0)),
                ocr_text=receipt_data.get("ocr_text")
            )

            return receipt
        except Exception as e:
            logger.error(f"Error creating Receipt object: {e}")
            raise


