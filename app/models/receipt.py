"""Data models for receipts and related information."""

from datetime import date
from typing import List, Optional
from pydantic import BaseModel, Field, validator, field_validator


class ReceiptItem(BaseModel):
    """A single item on a receipt"""

    name: str = Field(..., description="Item name or description")
    price: float = Field(..., description="Item price")
    quantity: float = Field(default=1.0, description="Item quantity")
    category: Optional[str] = Field(None, description="Expense category")

    @property
    def total_price(self) -> float:
        """Calculate the total price for the item"""
        return self.price * self.quantity


class Receipt(BaseModel):
    """Receipt data model"""

    vendor: str = Field(..., description="Vendor or store name")
    date: date = Field(..., description="Receipt date")
    items: List[ReceiptItem] = Field(default_factory=list, description="List of purchased items")
    subtotal: Optional[float] = Field(None, description="Subtotal amount before tax")
    tax: Optional[float] = Field(None, description="Tax amount")
    total: float = Field(..., description="Total amount")
    currency: str = Field(default="USD", description="Currency code")
    ocr_text: Optional[str] = Field(None, description="Raw OCR text from the receipt")
    image_path: Optional[str] = Field(None, description="Path to the original receipt image")

    @property
    def calculated_subtotal(self) -> float:
        """Calculate the subtotal from items."""
        return sum(item.total_price for item in self.items)

    @property
    def calculated_total(self) -> float:
        """Calculate the total including tax if available."""
        if self.tax is not None:
            return self.calculated_subtotal + self.tax
        return self.calculated_subtotal

    @field_validator("total")
    def validate_total(cls, v):
        """Validate that total is positive."""
        if v <= 0:
            raise ValueError("Total must be positive")
        return v

class ReceiptAnalysis(BaseModel):
    """Analysis results for a receipt."""

    receipt: Receipt
    correct_calculations: bool = Field(default=True, description="Whether the receipt calculations are correct")
    corrections: List[str] = Field(default_factory=list, description="List of corrections made")
    similar_receipts: List[dict] = Field(default_factory=list, description="Similar receipts found")
    expense_summary: Optional[dict] = Field(None, description="Summary of expenses by category")

    @property
    def has_corrections(self) -> bool:
        """Check if any corrections were made."""
        return len(self.corrections) > 0