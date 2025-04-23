"""Agent implementation for receipt analysis."""

import logging
from typing import Dict, List, Any, Optional, Callable

from llama_index.core.llms import LLM
from llama_index.llms.mistralai import MistralAI
from llama_index.core.agent import ReActAgent

from app.config import get_config
from app.agent.tools import ReceiptTools

logger = logging.getLogger(__name__)


class ReceiptAgent:
    """Agent for receipt analysis using LlamaIndex framework."""

    def __init__(
            self,
            tools: Optional[ReceiptTools] = None,
            llm: Optional[LLM] = None,
            verbose: bool = False
    ):
        """Initialize the receipt agent"""
        config = get_config()

        # Initialize tools
        self.tools = tools or ReceiptTools()

        # Initialize LLM for agent
        self.llm = llm or MistralAI(
            model="mistral-large-latest",
            api_key=config.mistral.api_key,
            temperature=0.1
        )

        # Create agent
        self.agent = ReActAgent.from_tools(
            self.tools.get_llamaindex_tools(),
            llm=self.llm,
            verbose=verbose
        )

        logger.info("Receipt agent initialized")

    async def analyze_receipt(self, image_path: str) -> Dict[str, Any]:
        """Analyze a receipt image using the agent and return the analysis results"""
        logger.info(f"Analyzing receipt image: {image_path}")

        # Create the task
        task = f"""
                Analyze the receipt image at path: {image_path}

                Follow these steps:
                1. Process the receipt image to extract structured data
                2. Validate the calculations to ensure accuracy
                3. Find similar receipts for context
                4. If the vendor is known, get vendor history

                Return the complete analysis with all extracted information.
                """
        try:
            # Execute the agent
            response = await self.agent.aquery(task)

            logger.info("Receipt analysis completed successfully")

            # Parse response and return result
            result = {
                "success": True,
                "analysis": response.response,
                "extracted_data": None,
                "validation": None,
                "similar_receipts": [],
                "vendor_history": None,
            }

            # Try to extract structured data from response
            if hasattr(response, 'metadata') and response.metadata:
                tool_outputs = response.metadata.get('tool_outputs', {})

                for output in tool_outputs:
                    if output.get('tool_name') == 'process_receipt_image':
                        result['extracted_data'] = output.get('output')
                    elif output.get('tool_name') == 'validate_calculations':
                        result['validation'] = output.get('output')
                    elif output.get('tool_name') == 'find_similar_receipts':
                        result['similar_receipts'] = output.get('output')
                    elif output.get('tool_name') == 'get_vendor_history':
                        result['vendor_history'] = output.get('output')

            return result

        except Exception as e:
            logger.error(f"Error analyzing receipt: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def process_receipt(self, image_path: str) -> Dict[str, Any]:
        """Process a receipt image (synchronous version) and return processed receipt data"""
        logger.info(f"Processing receipt image (direct): {image_path}")

        try:
            # Process the receipt directly using tools
            receipt_data = self.tools.process_receipt_image(image_path)

            # Validate calculations
            validation = self.tools.validate_calculations(receipt_data)

            # Find similar receipts
            similar_receipts = self.tools.find_similar_receipts(
                f"Vendor: {receipt_data.get('vendor', '')}"
            )

            # Get vendor history if available
            vendor_history = None
            vendor = receipt_data.get("vendor")
            if vendor:
                vendor_history = self.tools.get_vendor_history(vendor)

            # Combine results
            result = {
                "success": True,
                "receipt_data": receipt_data,
                "validation": validation,
                "similar_receipts": similar_receipts,
                "vendor_history": vendor_history
            }

            return result

        except Exception as e:
            logger.error(f"Error processing receipt: {e}")
            return {
                "success": False,
                "error": str(e)
            }



