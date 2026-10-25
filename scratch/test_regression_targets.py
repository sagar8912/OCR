import json
from src.engine.pipeline import OCREngine

def main():
    engine = OCREngine("models/best_model.pt")
    res = engine.process("sample/invoice_test.png", redact=False)

    words = [item["word"] for item in res["results"]]
    boxes = [item["bounding_box"] for item in res["results"]]
    field_types = [item.get("field_type") for item in res["results"]]

    print(f"Total Detected Regions: {len(words)}")

    targets = [
        "TECHNOLOGIES",
        "GURUGRAM",
        "HARYANA",
        "122001",
        "INV-2024-0156",
        "CUST-7890",
        "45,000",
        "90,000",
        "1,200",
        "6,000",
        "1,500",
        "7,500",
        "12,000",
        "24,000",
        "127,500",
        "22,950",
        "150,450",
        "9876543210",
        "HDFC0001234",
        "DESCRIPTION",
        "QUANTITY",
        "CONFIDENTIAL",
        "SUBTOTAL",
        "TOTAL",
        "AMOUNT",
    ]

    print("\n--- REGRESSION TARGET AUDIT ON INVOICE ---")
    matched_count = 0
    for target in targets:
        # Check exact or partial match in recognized words
        matches = [w for w in words if w == target or (len(target) > 4 and target in w) or (len(w) > 4 and w in target)]
        status = "EXACT MATCH" if target in words else ("CLOSE MATCH: " + str(matches) if matches else "NOT FOUND")
        if target in words:
            matched_count += 1
            print(f"  {target:16s} : PASS ({status})")
        else:
            print(f"  {target:16s} : {status}")

    print(f"\nExact matches found: {matched_count}/{len(targets)}")

if __name__ == "__main__":
    main()
