from src.engine.pipeline import OCREngine
engine = OCREngine("models/best_model.pt")
res = engine.process("sample/invoice_test.png", redact=False)
for i, item in enumerate(res["results"]):
    print(f"[{i:02d}] {item['word']:22s} | {item['bounding_box']} | {item.get('field_type')}")
