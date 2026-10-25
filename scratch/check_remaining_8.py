from src.engine.pipeline import OCREngine

engine = OCREngine("models/best_model.pt")
res = engine.process("sample/invoice_test.png", redact=False)

for item in res["results"]:
    b = item["bounding_box"]
    w = item["word"]
    raw = item.get("raw_word")
    ft = item.get("field_type")
    conf = item.get("confidence")

    # Check key regions
    # Invoice No: y in [80, 105], x > 800
    if 80 <= b[0] <= 105 and b[1] >= 800:
        print(f"INV region: Box={b} -> word='{w}' (raw='{raw}', ft={ft}, conf={conf})")
    # Customer ID: y in [130, 155], x > 800
    elif 130 <= b[0] <= 155 and b[1] >= 800:
        print(f"CUST region: Box={b} -> word='{w}' (raw='{raw}', ft={ft}, conf={conf})")
    # 1,500: y in [420, 445], x in [700, 760]
    elif 420 <= b[0] <= 445 and 700 <= b[1] <= 760:
        print(f"1,500 region: Box={b} -> word='{w}' (raw='{raw}', ft={ft}, conf={conf})")
    # 90,000: y in [350, 375], x in [910, 980]
    elif 350 <= b[0] <= 375 and 910 <= b[1] <= 980:
        print(f"90,000 region: Box={b} -> word='{w}' (raw='{raw}', ft={ft}, conf={conf})")
    # 22,950: y in [520, 545], x in [910, 980]
    elif 520 <= b[0] <= 545 and 910 <= b[1] <= 980:
        print(f"22,950 region: Box={b} -> word='{w}' (raw='{raw}', ft={ft}, conf={conf})")
    # 150,450: y in [555, 580], x in [900, 980]
    elif 555 <= b[0] <= 580 and 900 <= b[1] <= 980:
        print(f"150,450 region: Box={b} -> word='{w}' (raw='{raw}', ft={ft}, conf={conf})")
    # Account Number: y in [550, 575], x in [190, 310]
    elif 550 <= b[0] <= 575 and 190 <= b[1] <= 310:
        print(f"Account Number: Box={b} -> word='{w}' (raw='{raw}', ft={ft}, conf={conf})")
    # IFSC Code: y in [575, 600], x in [190, 320]
    elif 575 <= b[0] <= 600 and 190 <= b[1] <= 320:
        print(f"IFSC Code: Box={b} -> word='{w}' (raw='{raw}', ft={ft}, conf={conf})")
