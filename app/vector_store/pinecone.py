"""Pinecone vector store implementation"""

import logging
import uuid
import json
from typing import Dict, List, Any, Optional

import pinecone
from sentence_transformers import SentenceTransformer
from llama_index.core import Document
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import VectorStoreIndex, StorageContext

from app.config import get_config
from app.models.receipt import Receipt

logger = logging.getLogger(__name__)

class ReceiptVectorStore:
    def __init__(
            self,
            api_key: Optional[str] = None,
            environment: Optional[str] = None,
            index_name: str = "receipt-index",
        ):
        """Initialize pinecone vector store"""
        config = get_config()
        self.api_key = api_key or config.pinecone.api_key
        self.environment = environment or config.pinecone.environment
        self.index_name = index_name

        # Initialize Pinecone
        pinecone.init(api_key=self.api_key, environment=self.environment)

        # Get the Pinecone index
        self.index = pinecone.Index(self.index_name)

        # Initialize the embedding model
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Create LlamaIndex vector store and index
        self.vector_store = PineconeVectorStore(
            pinecone_index=self.index,
            environment=self.environment,
        )

        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        self.llama_index = VectorStoreIndex([], storage_context=storage_context)

        logger.info(f"Pinecone vector store initialized: {index_name}")

    def _receipt_to_text(self, receipt: Receipt) -> str:
        # Create a searchable text representation of the receipt
        lines = [
            f"Vendor: {receipt.vendor}",
            f"Date: {receipt.date}",
            f"Total: {receipt.total} {receipt.currency}",
            "Items:"
        ]

        for item in receipt.items:
            line = f"- {item.name}: {item.quantity} x {item.price} = {item.total_price}"
            if item.category:
                line += f" (Category: {item.category})"
            lines.append(line)

        if receipt.subtotal:
            lines.append(f"Subtotal: {receipt.subtotal}")

        if receipt.tax:
            lines.append(f"Tax: {receipt.tax}")

        lines.append(f"Total: {receipt.total}")

        if receipt.ocr_text:
            lines.append("\nOriginal Text:")
            lines.append(receipt.ocr_text)

        return "\n".join(lines)

    def add_receipt(self, receipt: Receipt) -> str:
        """Add a receipt to the vector store"""
        try:
            # Generate a unique ID for the receipt
            receipt_id = str(uuid.uuid4())

            # Convert receipt to text for embedding
            receipt_text = self._receipt_to_text(receipt)

            # Create a document with metadata
            metadata = {
                "receipt_id": receipt_id,
                "vendor": receipt.vendor,
                "date": str(receipt.date),
                "total": receipt.total,
                "currency": receipt.currency,
                "items_count": len(receipt.items),
                "receipt_json": json.dumps(receipt.model_dump(), default=str)
            }

            document = Document(
                text=receipt_text,
                metadata=metadata
            )

            # Add to the index
            self.llama_index.insert(document)

            logger.info(f"Added receipt to vector store with ID: {receipt_id}")
            return receipt_id

        except Exception as e:
            logger.error(f"Error adding receipt to vector store: {e}")
            raise

    def search_similar_receipts(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for similar receipts based on a query"""
        try:
            # Create a query engine
            query_engine = self.llama_index.as_query_engine(similarity_top_k=limit)

            # Execute the query
            response = query_engine.query(query)

            results = []
            if hasattr(response, 'source_nodes'):
                for node in response.source_nodes:
                    try:
                        # Extract the receipt JSON from metadata
                        receipt_json = node.metadata.get("receipt_json", "{}")
                        receipt_data = json.loads(receipt_json)

                        # Add similarity score and node text to the result
                        result = {
                            "receipt_id": node.metadata.get("receipt_id"),
                            "vendor": node.metadata.get("vendor"),
                            "date": node.metadata.get("date"),
                            "total": node.metadata.get("total"),
                            "receipt_data": receipt_data,
                            "score": node.score if hasattr(node, 'score') else None,
                        }
                        results.append(result)
                    except json.JSONDecodeError:
                        logger.warning(f"Error decoding receipt JSON from metadata")

            logger.info(f"Found {len(results)} similar receipts for query: {query}")
            return results

        except Exception as e:
            logger.error(f"Error searching similar receipts: {e}")
            return []

    def search_by_vendor(self, vendor_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for receipts from a specific vendor"""
        return self.search_similar_receipts(f"Vendor: {vendor_name}", limit)

    def get_receipt_by_id(self, receipt_id: str) -> Optional[Dict[str, Any]]:
        """Get a receipt by its ID"""
        try:
            # Create a query engine
            query_engine = self.llama_index.as_query_engine(similarity_top_k=10)

            # Execute a query that targets the specific receipt ID
            response = query_engine.query(f"receipt_id:{receipt_id}")

            if hasattr(response, 'source_nodes'):
                for node in response.source_nodes:
                    if node.metadata.get("receipt_id") == receipt_id:
                        receipt_json = node.metadata.get("receipt_json", "{}")
                        receipt_data = json.loads(receipt_json)
                        return {
                            "receipt_id": receipt_id,
                            "receipt_data": receipt_data,
                        }

            logger.warning(f"Receipt not found with ID: {receipt_id}")
            return None

        except Exception as e:
            logger.error(f"Error getting receipt by ID: {e}")
            return None


