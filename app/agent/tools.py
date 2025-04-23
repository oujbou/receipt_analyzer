"""Custom tools for the receipt analyzer agent"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from llama_index.core.tools import BaseTool, FunctionTool

from app.services.ocr import OCRService
from app.services.llm import LLMService
from app.vector_store.pinecone import ReceiptVectorStore
from app.models.receipt import Receipt

logger = logging.getLogger(__name__)


class ReceiptTools:
    """Collection of tools for receipt analysis and processing"""

    def __init__(
            self,
            ocr_service: Optional[OCRService] = None,
            llm_service: Optional[LLMService] = None,
            vector_store: Optional[ReceiptVectorStore] = None
    ):
        """Initialize receipt tools with required services"""

        self.ocr_service = ocr_service or OCRService()
        self.llm_service = llm_service or LLMService()
        self.vector_store = vector_store or ReceiptVectorStore()
        logger.info("Receipt tools initialized")

    def process_receipt_image(self, image_path: str) -> Dict[str, Any]:
        """Process a receipt image to extract  and return structured data"""
        logger.info(f"Processing receipt image: {image_path}")

        try:
            # Run OCR on the image
            ocr_result = self.ocr_service.process_image(image_path)

            # Extract text from OCR result
            extracted_text = ocr_result.get("text", "")

            # Use LLM to extract structured data
            receipt_data = self.llm_service.extract_receipt_data(extracted_text)

            # Classify expenses
            receipt_data = self.llm_service.classify_expenses(receipt_data)

            # Validate receipt data
            receipt_data = self.llm_service.validate_receipt(receipt_data)

            # Create Receipt object
            receipt = self.llm_service.create_receipt_object(receipt_data)

            # Add receipt to vector store
            receipt_id = self.vector_store.add_receipt(receipt)

            # Add receipt ID to result
            result = receipt_data.copy()
            result["receipt_id"] = receipt_id

            return result
        except Exception as e:
            logger.error(f"Error processing receipt image: {e}")
            raise

    def find_similar_receipts(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Find and return receipts similar to a query"""
        logger.info(f"Finding receipts similar to: {query}")
        try:
            results = self.vector_store.search_similar_receipts(query, limit=limit)
            return results
        except Exception as e:
            logger.error(f"Error finding similar receipts: {e}")
            return []

    def get_vendor_history(self, vendor_name: str) -> Dict[str, Any]:
        """Get history of receipts from a specific vendor, and return the vendor history data"""
        logger.info(f"Getting history for vendor: {vendor_name}")

        try:
            receipts = self.vector_store.search_by_vendor(vendor_name)

            # Calculate summary statistics
            total_spent = sum(float(r.get("total", 0)) for r in receipts)
            dates = [r.get("date") for r in receipts if r.get("date")]

            return {
                "vendor": vendor_name,
                "receipt_count": len(receipts),
                "total_spent": total_spent,
                "first_purchase": min(dates) if dates else None,
                "last_purchase": max(dates) if dates else None,
                "receipts": receipts
            }
        except Exception as e:
            logger.error(f"Error getting vendor history: {e}")
            return {
                "vendor": vendor_name,
                "receipt_count": 0,
                "receipts": []
            }

    def validate_calculations(self, receipt_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate receipt calculations for accuracy"""
        logger.info("Validating receipt calculations")

        try:
            # Calculate total from items
            items = receipt_data.get("items", [])
            calculated_total = sum(
                item.get("price", 0) * item.get("quantity", 1) for item in items
            )

            # Get receipt total
            receipt_total = float(receipt_data.get("total", 0))

            # Check for tax
            tax = receipt_data.get("tax")
            if tax is not None:
                calculated_total += float(tax)

            # Calculate difference
            difference = abs(calculated_total - receipt_total)

            # Check if totals match (with small tolerance for floating point errors)
            is_valid = difference < 0.01

            validation_result = {
                "is_valid": is_valid,
                "calculated_total": calculated_total,
                "receipt_total": receipt_total,
                "difference": difference,
                "corrections": []
            }

            # Add correction if needed
            if not is_valid:
                validation_result["corrections"].append(
                    f"Total mismatch: Receipt shows {receipt_total} but calculated total is {calculated_total}."
                )

            return validation_result
        except Exception as e:
            logger.error(f"Error validating calculations: {e}")
            return {
                "is_valid": False,
                "error": str(e),
                "corrections": [f"Error during validation: {str(e)}"]
            }

    def get_llamaindex_tools(self) -> List[BaseTool]:
        """Get tools for use with LlamaIndex agent"""
        return [
            FunctionTool.from_defaults(
                name="process_receipt_image",
                description="Process a receipt image to extract structured data",
                fn=self.process_receipt_image,
            ),
            FunctionTool.from_defaults(
                name="find_similar_receipts",
                description="Find receipts similar to a query",
                fn=self.find_similar_receipts,
            ),
            FunctionTool.from_defaults(
                name="get_vendor_history",
                description="Get history of receipts from a specific vendor",
                fn=self.get_vendor_history,
            ),
            FunctionTool.from_defaults(
                name="validate_calculations",
                description="Validate receipt calculations for accuracy",
                fn=self.validate_calculations,
            ),
        ]


