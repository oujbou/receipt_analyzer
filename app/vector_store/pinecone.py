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

        # Get the Pinecone index (assuming it already exists)
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

        logger.info(f"Pinecone vector store initialized with existing index: {index_name}")

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


