import cv2
from src.engine.pipeline import OCREngine

p = "C:/Users/HP/.gemini/antigravity-ide/brain/a5d7d016-27af-4054-bacf-b790cae60c2b/.user_uploaded/media_1790262533853.png"
engine = OCREngine("models/best_model.pt")
res = engine.process(p, redact=False)
print("Results count on media_1790262533853.png:", len(res["results"]))
for i, item in enumerate(res["results"]):
    w = item["word"]
    box = item["bounding_box"]
    ft = item.get("field_type")
    print(f"[{i:03d}] {w:22s} | {box} | {ft}")
