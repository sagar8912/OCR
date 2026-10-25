import cv2
import numpy as np
from src.engine.pipeline import OCREngine
from src.engine.decoder import analyze_crop_topology

engine = OCREngine("models/best_model.pt")
crop = cv2.imread("scratch/crop_inv_full.png")
topo = analyze_crop_topology(crop)
print(f"Topology on INV crop: has_hyphen={topo['has_hyphen']}, hyphens={topo['hyphens']}")

res = engine.recognizer.decoder.decode_identifier(None, crop=crop, raw_text="N404Y0SG")
print(f"Identifier decode result: {res}")
