"""Automated tests for Numeric and Alphanumeric OCR decoding.

Covers all 10 required test categories:
1. Pure numbers
2. Numbers with commas
3. Decimal values
4. Dates
5. Invoice IDs
6. Customer IDs
7. IFSC-like alphanumeric strings
8. English words
9. Mixed word + number
10. Unknown identifiers
"""

import pytest
import numpy as np
import torch
from PIL import Image, ImageDraw

from src.engine.decoder import LexiconDecoder, FieldType, classify_field_type
from src.engine.pipeline import OCREngine


@pytest.fixture(scope="module")
def engine():
    return OCREngine(model_path="models/best_model.pt", device="cpu")


@pytest.fixture(scope="module")
def decoder():
    return LexiconDecoder()


def create_synthetic_text_crop(text: str, height: int = 32, pad: int = 10) -> np.ndarray:
    """Create a clean synthetic text crop for OCR verification."""
    # Approximate width based on length
    width = max(64, len(text) * 16 + pad * 2)
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Default PIL font
    draw.text((pad, 8), text, fill=(0, 0, 0))
    return np.array(img)


# 1. Pure numbers
def test_pure_numbers(decoder, engine):
    samples = ["123", "4567", "9876543210", "122001", "24", "2026"]
    for s in samples:
        ftype = classify_field_type(s)
        assert ftype == FieldType.NUMBER
        res = decoder.decode(torch.randn(32, 1, 37), raw_text=s)
        assert res["word"] == s
        assert res["field_type"] == FieldType.NUMBER.value
        assert res["method"] == "NUMERIC_PRESERVED"


# 2. Numbers with commas
def test_numbers_with_commas(decoder):
    samples = ["45,000", "90,000", "1,200", "6,000", "150,450", "1,000,000"]
    for s in samples:
        ftype = classify_field_type(s)
        assert ftype in [FieldType.CURRENCY, FieldType.NUMBER]
        res = decoder.decode(torch.randn(32, 1, 37), raw_text=s)
        assert res["word"] == s
        assert res["method"] == "NUMERIC_PRESERVED"


# 3. Decimal values
def test_decimal_values(decoder):
    samples = ["19.99", "450.50", "0.05", "127.50"]
    for s in samples:
        ftype = classify_field_type(s)
        assert ftype in [FieldType.NUMBER, FieldType.CURRENCY]
        res = decoder.decode(torch.randn(32, 1, 37), raw_text=s)
        assert res["word"] == s
        assert res["method"] == "NUMERIC_PRESERVED"


# 4. Dates
def test_dates(decoder):
    samples = ["24/09/2026", "15-08-2024", "2024-01-15", "24 Sep 2026"]
    for s in samples:
        ftype = classify_field_type(s)
        assert ftype == FieldType.DATE
        res = decoder.decode(torch.randn(32, 1, 37), raw_text=s)
        assert res["word"] == s
        assert res["field_type"] == FieldType.DATE.value


# 5. Invoice IDs
def test_invoice_ids(decoder):
    samples = ["INV-2024-0156", "INV-2023-8901", "INV-2026-001"]
    for s in samples:
        ftype = classify_field_type(s)
        assert ftype == FieldType.ALPHANUMERIC_ID
        res = decoder.decode(torch.randn(32, 1, 37), raw_text=s)
        assert res["word"] == s
        assert res["field_type"] == FieldType.ALPHANUMERIC_ID.value


# 6. Customer IDs
def test_customer_ids(decoder):
    samples = ["CUST-7890", "CUST-1245", "CUST-9921"]
    for s in samples:
        ftype = classify_field_type(s)
        assert ftype == FieldType.ALPHANUMERIC_ID
        res = decoder.decode(torch.randn(32, 1, 37), raw_text=s)
        assert res["word"] == s
        assert res["field_type"] == FieldType.ALPHANUMERIC_ID.value


# 7. IFSC-like alphanumeric strings
def test_ifsc_alphanumeric_strings(decoder):
    samples = ["HDFC0001234", "SBIN0004567", "ICIC0009876", "UTIB0001122"]
    for s in samples:
        ftype = classify_field_type(s)
        assert ftype == FieldType.ALPHANUMERIC_ID
        res = decoder.decode(torch.randn(32, 1, 37), raw_text=s)
        assert res["word"] == s
        assert res["field_type"] == FieldType.ALPHANUMERIC_ID.value


# 8. English words
def test_english_words(decoder):
    samples = ["INVOICE", "CONFIDENTIAL", "DESCRIPTION", "QUANTITY", "SUBTOTAL"]
    for s in samples:
        ftype = classify_field_type(s)
        assert ftype == FieldType.WORD
        assert s in decoder.words_set


# 9. Mixed word + number
def test_mixed_word_number(decoder):
    samples = ["Sector 62", "Monitor 24 Inch", "Tax 18%"]
    for s in samples:
        # Should not be corrupted to single dictionary word
        res = decoder.decode(torch.randn(32, 1, 37), raw_text=s)
        assert res["word"] == s


# 10. Unknown identifiers
def test_unknown_identifiers(decoder):
    samples = ["TXN-98421", "REF#88120", "BATCH-2024A"]
    for s in samples:
        res = decoder.decode(torch.randn(32, 1, 37), raw_text=s)
        # Must retain original identifier structure and never force dictionary word
        assert res["word"] == s
        assert res["method"] == "NUMERIC_PRESERVED"
