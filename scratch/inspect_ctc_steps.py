"""Inspect CTC time-step logits for invoice crops."""
import cv2
import torch
import torch.nn.functional as F
from src.engine.recognizer import TextRecognizer
from src.training.dataset import DEFAULT_CONVERTER

recognizer = TextRecognizer(model_path="models/best_model.pt", device="cpu")
img = cv2.imread("sample/invoice_test.png")

crops = {
    "INVOICE": img[22:62, 715:925],
    "DESCRIPTION": img[315:338, 250:365],
    "QUANTITY": img[315:338, 485:575],
    "TOTAL": img[555:578, 610:680],
    "AMOUNT": img[555:578, 685:780],
}

print("=" * 60)
for name, crop in crops.items():
    t = recognizer.preprocessor.prepare_for_ocr(crop)
    with torch.no_grad():
        log_probs = recognizer.model(t) # (T, 1, 37)
        probs = F.softmax(log_probs, dim=-1)
        best_indices = torch.argmax(log_probs, dim=-1).squeeze(1).numpy()
    
    seq_str = []
    top_tokens = []
    for step, idx in enumerate(best_indices):
        char = DEFAULT_CONVERTER.idx2char.get(idx, "_")
        prob = probs[step, 0, idx].item()
        seq_str.append(char)
        if char != "_":
            top_tokens.append(f"{char}({prob:.2f})")
    
    decoded = DEFAULT_CONVERTER.decode_greedy(best_indices)
    print(f"\nTarget: {name}")
    print(f"Decoded: '{decoded}'")
    print(f"Time steps: {''.join(seq_str)}")
    print(f"Non-blank tokens: {' '.join(top_tokens)}")
