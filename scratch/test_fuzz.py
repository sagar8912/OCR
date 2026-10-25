"""Test RapidFuzz correction on predicted OCR tokens."""
from rapidfuzz import process, fuzz

lexicon = [
    "INVOICE", "DESCRIPTION", "QUANTITY", "UNIT", "PRICE", "AMOUNT",
    "SUBTOTAL", "TAX", "TOTAL", "ITEM", "NO", "CUSTOMER", "ID", "DATE",
    "BILL", "TO", "PAYMENT", "DETAILS", "ACCOUNT", "NAME", "NUMBER",
    "IFSC", "CODE", "BANK", "LAPTOP", "DELL", "INSPIRON", "WIRELESS",
    "MOUSE", "KEYBOARD", "MONITOR", "INCH", "CONFIDENTIAL", "NOTE",
    "SIGNATURE", "BUSINESS", "PARK", "SECTOR", "ABC", "TECHNOLOGIES",
    "PVT", "LTD", "CORPORATION", "INNOVATION", "DRIVE", "HYDERABAD",
    "TELANGANA", "INDIA", "GURUGRAM", "HARYANA", "HDFC"
]

test_preds = [
    "INOCE", "DESCTION", "OUANTTY", "OANTTY", "ANOUNT", "SULTOAA",
    "BLL", "AAC", "FSCCOOE", "HURCCANK", "CEOOAD", "MEESMOUS",
    "45000", "90000", "INV-2024-0156", "2", "5"
]

print("RapidFuzz Vocabulary Correction Test:")
for p in test_preds:
    match, score, _ = process.extractOne(p, lexicon, scorer=fuzz.ratio)
    if score >= 70 and not any(c.isdigit() for c in p):
        print(f"Pred: {p:12s} -> Matched: {match:15s} (Score: {score:.1f})")
    else:
        print(f"Pred: {p:12s} -> Retained: {p:15s} (Score: {score:.1f}, numeric or low match)")
