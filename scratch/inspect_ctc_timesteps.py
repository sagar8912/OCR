import cv2
import torch
from src.engine.pipeline import OCREngine

engine = OCREngine("models/best_model.pt")
img = cv2.imread("sample/invoice_test.png")
crop = img[561:578, 907:974]
p1 = crop[:, 0:30]
p2 = crop[:, 33:]

conv = engine.recognizer.converter

for name, patch in [("P1 (150)", p1), ("P2 (450)", p2)]:
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    w = max(96, patch.shape[1] * 4)
    resized = cv2.resize(gray, (w, 32), interpolation=cv2.INTER_AREA)
    norm = (resized.astype(float) / 127.5) - 1.0
    tensor = torch.from_numpy(norm).float().unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        lp = engine.recognizer.model(tensor)
    
    probs = torch.softmax(lp[:, 0], dim=-1)
    T = probs.shape[0]
    print(f"\n--- {name} (T={T}) ---")
    for t in range(T):
        top3_vals, top3_idx = torch.topk(probs[t], 3)
        chars = [conv.idx2char.get(i.item(), "[BLANK]") for i in top3_idx]
        vals = [f"{v.item():.2f}" for v in top3_vals]
        if chars[0] != "[BLANK]" or float(vals[0]) < 0.95:
            print(f"t={t:02d}: {chars[0]}({vals[0]}) | {chars[1]}({vals[1]}) | {chars[2]}({vals[2]})")
