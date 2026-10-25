import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from rapidfuzz.distance import Levenshtein

from src.engine.recognizer import TextRecognizer
from src.training.dataset import DEFAULT_CONVERTER

# Load model
rec = TextRecognizer(model_path="models/best_model.pt")

# Load Lexicon
with open("IIIT5K-Word_V3.0/IIIT5K/lexicon.txt", "r", encoding="utf-8", errors="ignore") as f:
    lexicon_words = [line.strip().upper() for line in f if line.strip()]

words_set = set(lexicon_words)
by_len = defaultdict(list)
for w in lexicon_words:
    by_len[len(w)].append(w)

print(f"Loaded {len(lexicon_words)} lexicon words.")

def score_word_ctc(log_probs_3d, word):
    # log_probs_3d: (T, 1, C)
    encoded = DEFAULT_CONVERTER.encode(word)
    if not encoded:
        return -999.0
    target = torch.tensor(encoded, dtype=torch.long)
    inp_len = torch.tensor([log_probs_3d.size(0)], dtype=torch.long)
    tgt_len = torch.tensor([len(target)], dtype=torch.long)
    loss = F.ctc_loss(log_probs_3d, target, inp_len, tgt_len, blank=0, reduction="none")
    return -loss.item()

def decode_crop_with_lexicon(crop, max_edit_dist=2):
    t = rec.preprocessor.prepare_for_ocr(crop)
    with torch.no_grad():
        log_probs = rec.model(t) # (T, 1, C)
    
    # Greedy decoding
    best_idx = torch.argmax(log_probs, dim=-1)[:, 0].numpy()
    raw_word = rec.converter.decode_greedy(best_idx)
    
    # If contains digits or special chars, preserve strictly
    has_digit = any(c.isdigit() for c in raw_word)
    if has_digit or len(raw_word) <= 1:
        return raw_word, raw_word, "NUMERIC_PRESERVED"
    
    # If already an exact dictionary word
    if raw_word in words_set:
        return raw_word, raw_word, "EXACT_LEXICON"
    
    # Find candidate words near raw_word
    L = len(raw_word)
    max_d = 1 if L <= 4 else (2 if L <= 8 else 3)
    
    candidates = []
    for l in range(max(2, L - max_d), L + max_d + 1):
        for w in by_len[l]:
            # Quick 1st char check or length heuristic
            if abs(len(w) - L) <= max_d:
                # Fast edit distance check
                d = Levenshtein.distance(raw_word, w)
                if d <= max_d:
                    candidates.append((w, d))
    
    if not candidates:
        return raw_word, raw_word, "NO_CANDIDATE"
    
    # Sort candidates by edit distance and evaluate top candidates with CTC
    candidates.sort(key=lambda x: x[1])
    top_cands = candidates[:15]
    
    raw_ctc = score_word_ctc(log_probs, raw_word)
    
    best_cand = raw_word
    best_score = -float("inf")
    
    for cand, d in top_cands:
        ctc_lp = score_word_ctc(log_probs, cand)
        # Score combining acoustic CTC likelihood + edit distance penalty
        norm_d = d / max(len(cand), len(raw_word))
        # Each edit distance cost ~ 3.0 in log-prob
        combined_score = ctc_lp - 3.0 * norm_d
        
        if combined_score > best_score:
            best_score = combined_score
            best_cand = cand
            
    return raw_word, best_cand, f"CORRECTED(dist={Levenshtein.distance(raw_word, best_cand)})"

# Test samples
test_cases = [
    ("IIIT5K-Word_V3.0/IIIT5K/test/1002_1.png", "PRIVATE"),
    ("IIIT5K-Word_V3.0/IIIT5K/test/1104_2.png", "ATTENTION"),
    ("IIIT5K-Word_V3.0/IIIT5K/test/321_14.png", "MAIL"),
    ("IIIT5K-Word_V3.0/IIIT5K/test/1016_1.png", "STATE"),
    ("IIIT5K-Word_V3.0/IIIT5K/test/134_3.png", "BABY"),
]

print("\n--- IIIT-5K NEAR-MISS TEST ---")
for path, expected in test_cases:
    im = cv2.imread(path)
    raw, decoded, status = decode_crop_with_lexicon(im)
    print(f"Expected: {expected:<12} | Raw: {raw:<12} | Decoded: {decoded:<12} | Status: {status}")
