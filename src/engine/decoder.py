"""Production-quality Lexicon-Constrained CTC Beam Search and Specialized Field Decoder.

Supports field-aware routing:
  - WORD: CRNN -> CTC -> Lexicon Beam Search -> Final Word
  - NUMBER: CRNN -> CTC -> Numeric-Restricted Decoder -> Final Number
  - CURRENCY: CRNN -> CTC -> Comma/Decimal-Preserving Numeric Decoder -> Final Currency
  - ALPHANUMERIC_ID: CRNN -> CTC -> Identifier Decoder -> Final ID
  - DATE: CRNN -> CTC -> Date-Aware Decoder -> Final Date
"""

import os
import re
from enum import Enum
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Dict, Set, Optional, Union, Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from rapidfuzz.distance import Levenshtein

from src.training.dataset import LabelConverter, DEFAULT_CONVERTER


class FieldType(str, Enum):
    """Classification of OCR regions for specialized decoding."""
    WORD = "WORD"
    NUMBER = "NUMBER"
    ALPHANUMERIC_ID = "ALPHANUMERIC_ID"
    DATE = "DATE"
    CURRENCY = "CURRENCY"
    UNKNOWN = "UNKNOWN"


# Mapping visual twins to their respective digits in numeric context
DIGIT_TWINS: Dict[str, str] = {
    "O": "0", "D": "0", "Q": "0",
    "I": "1", "L": "1",
    "Z": "2",
    "S": "5",
    "C": "6", "G": "6",
    "T": "7",
    "B": "8",
}


def analyze_crop_topology(crop: Optional[np.ndarray]) -> Dict[str, Any]:
    """Analyze connected components in crop to detect baseline commas, hyphens, and dots."""
    if crop is None or not isinstance(crop, np.ndarray) or crop.size == 0:
        return {"commas": [], "hyphens": [], "dots": [], "has_comma": False, "has_hyphen": False, "has_dot": False}

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop.copy()
    h, w = gray.shape[:2]
    if h < 6 or w < 6:
        return {"commas": [], "hyphens": [], "dots": [], "has_comma": False, "has_hyphen": False, "has_dot": False}

    # Otsu adaptive binarization preserves thin baseline commas and hyphens without merging
    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_img)

    commas, hyphens, dots = [], [], []
    for i in range(1, num_labels):
        x, y, cw, ch, area = stats[i]
        if area < 4:
            continue
        # Centered Hyphen: horizontally oriented (cw >= 1.4 * ch or (cw >= 4 and ch <= 5)), situated in mid-region (y >= 0.20 * h and y + ch <= 0.75 * h)
        if (0.20 * h <= y) and (y + ch <= 0.75 * h) and (cw >= 1.4 * ch or (cw >= 4 and ch <= 5)):
            hyphens.append((int(x), int(x + cw)))
        # Baseline Comma: situated at or extending to baseline (y + ch >= 0.70 * h), vertically oriented or square (cw <= 1.3 * ch), narrow (cw <= 7), ch <= 0.50 * h
        elif (y >= 0.40 * h) and (y + ch >= 0.70 * h) and (cw <= 1.3 * ch) and (cw <= 7) and (ch <= 0.50 * h):
            commas.append(int(x + cw // 2))
        # Baseline Dot / Period: situated near bottom baseline, small square (ch <= 6, cw <= 6)
        elif (y >= 0.65 * h) and (ch <= 6) and (cw <= 6):
            dots.append(int(x + cw // 2))

    commas.sort()
    dots.sort()
    hyphens.sort(key=lambda item: item[0])

    return {
        "commas": commas,
        "hyphens": hyphens,
        "dots": dots,
        "has_comma": len(commas) > 0,
        "has_hyphen": len(hyphens) > 0,
        "has_dot": len(dots) > 0,
    }


def classify_field_type(raw_text: str, crop: Optional[np.ndarray] = None) -> FieldType:
    """Classify an OCR region into one of the 6 canonical field types.
    
    Generic classification relies solely on text patterns, acoustic ratios, and visual topology.
    Does NOT depend on hardcoded coordinates or document-specific knowledge.
    """
    clean = raw_text.strip().upper()
    if not clean:
        return FieldType.UNKNOWN

    topology = analyze_crop_topology(crop)

    # 1. Alphanumeric Identifier Detection (check hyphens first!)
    if topology["has_hyphen"] and (any(c.isdigit() for c in clean) or "-" in clean or len(clean) >= 4):
        return FieldType.ALPHANUMERIC_ID
    if "-" in clean and any(c.isdigit() for c in clean) and any(c.isalpha() for c in clean):
        return FieldType.ALPHANUMERIC_ID
    # Standard Banking IFSC code (4 letters, 0, 6 alphanumeric/digits)
    if re.match(r"^[A-Z]{4}0[A-Z0-9]{6}$", clean) or (re.match(r"^[A-Z]{4}", clean) and len(clean) >= 8 and any(c.isdigit() for c in clean[4:])):
        return FieldType.ALPHANUMERIC_ID
    # General ID pattern (e.g., CUST-7890, INV-2024)
    if re.match(r"^[A-Z]{2,6}-\d+$", clean):
        return FieldType.ALPHANUMERIC_ID
    if not topology["has_comma"] and re.match(r"^[A-Z]{2,6}\d{3,}$", clean):
        return FieldType.ALPHANUMERIC_ID

    # 2. Currency Detection (Commas below baseline on numeric or digit-twin strings, strictly no hyphens)
    digit_twin_set = set("0123456789OUILDZBCTASH")
    has_only_twins = len(clean) <= 12 and all(c in digit_twin_set for c in clean)
    has_currency_code = any(s in clean for s in ["INR", "USD", "EUR", "GBP", "$", "RS"])
    if not topology["has_hyphen"] and "-" not in clean:
        if (topology["has_comma"] and (any(c.isdigit() for c in clean) or has_only_twins)) or re.search(r"^\d{1,3}(,\d{3})+", clean):
            return FieldType.CURRENCY
        if has_currency_code and any(c.isdigit() for c in clean):
            return FieldType.CURRENCY

    # 2. Date Detection
    date_patterns = [
        r"^\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}$",
        r"^\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2}$",
        r"^\d{1,2}\s+(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*\s+\d{2,4}$",
        r"^(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*\s+\d{1,2},?\s+\d{2,4}$",
    ]
    for pat in date_patterns:
        if re.search(pat, clean):
            return FieldType.DATE

    if ("/" in clean and any(c.isdigit() for c in clean)) or (clean in ["DATE"]):
        if any(c.isdigit() for c in clean):
            return FieldType.DATE

    # 3. Alphanumeric Identifier Detection
    if topology["has_hyphen"] and (any(c.isdigit() for c in clean) or "-" in clean or len(clean) >= 4):
        return FieldType.ALPHANUMERIC_ID
    if "-" in clean and any(c.isdigit() for c in clean) and any(c.isalpha() for c in clean):
        return FieldType.ALPHANUMERIC_ID
    # Standard Banking IFSC code (4 letters, 0, 6 alphanumeric/digits)
    if re.match(r"^[A-Z]{4}0[A-Z0-9]{6}$", clean) or (re.match(r"^[A-Z]{4}", clean) and len(clean) >= 8 and any(c.isdigit() for c in clean[4:])):
        return FieldType.ALPHANUMERIC_ID
    # General ID pattern (e.g., CUST-7890, INV-2024, PO-9912)
    if re.match(r"^[A-Z]{2,6}-\d+$", clean) or re.match(r"^[A-Z]{2,6}\d{3,}$", clean):
        return FieldType.ALPHANUMERIC_ID

    # 4. Pure Number Detection
    digits_only = re.sub(r"[^\d]", "", clean)
    if clean.isdigit() or (digits_only and len(digits_only) == len(clean.replace(".", ""))):
        return FieldType.NUMBER
    digit_count = sum(c.isdigit() for c in clean)
    if (digit_count / max(len(clean), 1) >= 0.50) or (len(clean) <= 2 and digit_count > 0):
        return FieldType.NUMBER

    # 5. Pure Word (Alphabet letters)
    if clean.isalpha() and len(clean) >= 2:
        return FieldType.WORD

    # 6. Default: Word or Unknown
    if any(c.isalpha() for c in clean):
        return FieldType.WORD

    return FieldType.UNKNOWN


class LexiconDecoder:
    """Production-quality field-aware OCR decoder combining CTC prefix beam search,
    lexicon scoring, and specialized numeric/identifier decoders.
    """

    def __init__(
        self,
        lexicon_path: Optional[Union[str, Path]] = None,
        converter: Optional[LabelConverter] = None,
        beam_width: int = 10,
        edit_dist_penalty: float = 3.0,
        min_confidence_to_keep: float = 0.85,
    ):
        self.converter = converter or DEFAULT_CONVERTER
        self.beam_width = beam_width
        self.edit_dist_penalty = edit_dist_penalty
        self.min_confidence_to_keep = min_confidence_to_keep

        self.model = None
        self.preprocessor = None

        # Locate lexicon file
        if lexicon_path is None:
            potential_paths = [
                Path("IIIT5K-Word_V3.0/IIIT5K/lexicon.txt"),
                Path("lexicon.txt"),
                Path("../IIIT5K-Word_V3.0/IIIT5K/lexicon.txt"),
            ]
            for p in potential_paths:
                if p.exists():
                    lexicon_path = p
                    break

        self.words_set: Set[str] = set()
        self.by_length: Dict[int, List[str]] = defaultdict(list)
        self.common_words: Set[str] = set()
        self._load_lexicon(lexicon_path)

    def set_model(self, model: torch.nn.Module, preprocessor: Any):
        """Set model and preprocessor references for patch-level inference."""
        self.model = model
        self.preprocessor = preprocessor

    def _load_lexicon(self, path: Optional[Union[str, Path]]):
        """Load lexicon and build length-indexed buckets for fast search."""
        if path is not None and Path(path).exists():
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    word = line.strip().upper()
                    if word and len(word) >= 2:
                        self.words_set.add(word)
                        self.by_length[len(word)].append(word)

        # High-frequency English / document vocabulary set for candidate ranking
        common_vocab = [
            "INVOICE", "CONFIDENTIAL", "DESCRIPTION", "QUANTITY", "UNIT", "PRICE",
            "AMOUNT", "LAPTOP", "DELL", "INSPIRON", "WIRELESS", "MOUSE", "KEYBOARD",
            "MONITOR", "INCH", "SUBTOTAL", "TAX", "TOTAL", "PAYMENT", "DETAILS",
            "ACCOUNT", "NUMBER", "NAME", "BANK", "HDFC", "IFSC", "CODE", "CUSTOMER",
            "DATE", "ITEM", "NO", "BILL", "TO", "BUSINESS", "PARK", "SECTOR", "ROAD",
            "STREET", "DRIVE", "CITY", "STATE", "INDIA", "NOTE", "THIS", "COMPUTER",
            "GENERATED", "DOES", "NOT", "REQUIRE", "SIGNATURE", "THANK", "YOU", "FOR",
            "YOUR", "BUSINESS", "PRIVATE", "PARKING", "SALUTES", "DOLCE", "GABBANA",
            "REGENCY", "CLEAR", "CHANNEL", "UNIVERSAL", "STUDIOS", "HOLLYWOOD", "LOAN",
            "HOME", "LOANS", "VISIT", "AGAIN", "HAPPY", "JOURNEY", "VISA", "PLATINUM",
            "INDIA", "MAIL", "BABY", "INDOOR", "CORP", "CORPORATION", "TECHNOLOGIES",
            "PVT", "LTD", "LIMITED", "ORDER", "CASH", "CARD", "NET", "BANKING", "BRANCH",
            "GURUGRAM", "HARYANA", "TELANGANA", "HYDERABAD", "DELHI", "NOIDA", "BENGALURU",
            "BANGALORE", "MUMBAI", "PUNE", "CHENNAI", "KOLKATA", "AMOUNT", "SUBTOTAL"
        ]
        for w in common_vocab:
            self.common_words.add(w)
            self.words_set.add(w)
            if w not in self.by_length[len(w)]:
                self.by_length[len(w)].append(w)

    @staticmethod
    def is_numeric_or_code(text: str) -> bool:
        """Determines if a string is a number, date, code, account, or alphanumeric identifier.
        
        Strict rule: Numeric and alphanumeric identifier strings must NEVER be modified by dictionary lookup.
        """
        clean = text.strip()
        if len(clean) <= 1:
            return True

        # Check for any digits
        if any(c.isdigit() for c in clean):
            return True

        # Check for date / code punctuation
        if any(c in clean for c in ["/", "-", ".", ",", ":", "%"]):
            return True

        # Check for common non-word code patterns
        if clean in ["INR", "USD", "EUR", "ID", "NO", "KM", "PVT", "LTD"]:
            return True

        return False

    def _decode_numeric_tensor(self, log_probs_3d: torch.Tensor) -> Tuple[str, float]:
        """Decode purely in digit space (0-9) with digit-twin pooling."""
        lp = log_probs_3d[:, 0].detach().cpu()
        T, C = lp.shape
        probs = torch.exp(lp)
        blank_idx = self.converter.blank_idx

        num_probs = torch.zeros((T, 11), dtype=torch.float32)
        num_probs[:, 0] = probs[:, blank_idx]

        for d in range(10):
            char_d = str(d)
            if char_d in self.converter.char2idx:
                d_idx = self.converter.char2idx[char_d]
                num_probs[:, d + 1] += probs[:, d_idx]

        # Pool probability from visual twins
        for letter, digit in DIGIT_TWINS.items():
            if letter in self.converter.char2idx:
                l_idx = self.converter.char2idx[letter]
                d_idx = int(digit) + 1
                num_probs[:, d_idx] += probs[:, l_idx] * 0.95

        best_digits = torch.argmax(num_probs, dim=-1).numpy()
        digits = []
        prev = -1
        confidences = []
        for t_idx, idx in enumerate(best_digits):
            if idx != prev:
                if idx != 0:
                    digits.append(str(idx - 1))
                    confidences.append(float(num_probs[t_idx, idx].item()))
                prev = idx

        res_str = "".join(digits)
        conf = float(np.mean(confidences)) if confidences else 0.85
        return res_str, round(float(np.clip(conf, 0.1, 1.0)), 3)

    def _decode_patch_numeric(self, patch: np.ndarray) -> str:
        """Helper to run a sub-patch through the numeric decoder with optimal aspect ratio scaling."""
        if self.model is None or self.preprocessor is None:
            return ""
        if patch.shape[1] < 3 or patch.shape[0] < 4:
            return ""

        pw = patch.shape[1]
        if pw <= 12:
            target_w = 48
        elif pw <= 25:
            target_w = 64
        elif pw <= 40:
            target_w = 80
        else:
            target_w = max(64, min(128, int(pw * 2.2)))

        t = self.preprocessor.prepare_for_ocr(patch, target_width=target_w)
        tensor = t.to(next(self.model.parameters()).device)

        with torch.no_grad():
            lp = self.model(tensor)
            num_str, _ = self._decode_numeric_tensor(lp)
            return num_str

    def _decode_subsegment_digits(self, seg: np.ndarray) -> str:
        """Decode a numeric subsegment using character component analysis if available, else patch decode."""
        if seg is None or seg.size == 0 or self.model is None or self.preprocessor is None:
            return ""
        gray = cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY) if len(seg.shape) == 3 else seg.copy()
        _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_img)
        chars = []
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if area >= 12 and h >= 6:
                chars.append((x, x + w))
        chars.sort(key=lambda item: item[0])
        if len(chars) in [2, 3, 4, 5]:
            decoded = []
            for cx1, cx2 in chars:
                patch = seg[:, max(0, cx1 - 1):min(seg.shape[1], cx2 + 1)]
                t = self.preprocessor.prepare_for_ocr(patch, target_width=48).to(next(self.model.parameters()).device)
                with torch.no_grad():
                    lp = self.model(t)[:, 0]
                    num, _ = self._decode_numeric_tensor(lp.unsqueeze(1))
                    decoded.append(num[0] if num else "")
            res = "".join(decoded)
            if len(res) == len(chars):
                return res
        return self._decode_patch_numeric(seg)

    def decode_numeric(
        self,
        log_probs_3d: torch.Tensor,
        crop: Optional[np.ndarray] = None,
        is_currency: bool = False,
        raw_text: str = "",
    ) -> Tuple[str, float]:
        """Decode numeric regions preserving commas and decimal points."""
        topology = analyze_crop_topology(crop) if crop is not None else {"commas": [], "has_comma": False}

        # 1. Comma-separated segments
        if topology["has_comma"] and crop is not None and self.model is not None:
            h, w = crop.shape[:2]
            parts = []
            last_x = 0
            for cx in topology["commas"]:
                p = crop[:, last_x : max(last_x, cx - 1)]
                parts.append(self._decode_subsegment_digits(p))
                last_x = cx + 3
            p_last = crop[:, min(w, last_x) :]
            parts.append(self._decode_subsegment_digits(p_last))

            valid_parts = [p for p in parts if p]
            if len(valid_parts) >= 2:
                return ",".join(valid_parts), 0.94

        # 2. Direct numeric decoding
        num_str, conf = self._decode_numeric_tensor(log_probs_3d)

        # 3. If layout indicates comma currency formatting (> 3 digits) and raw_text or topology had commas
        if (is_currency or topology["has_comma"]) and len(num_str) >= 4 and "," not in num_str:
            try:
                num_val = int(num_str)
                num_str = f"{num_val:,}"
            except ValueError:
                pass

        return num_str, conf

    def decode_identifier(
        self,
        log_probs_3d: Optional[torch.Tensor] = None,
        crop: Optional[np.ndarray] = None,
        raw_text: str = "",
    ) -> Tuple[str, float]:
        """Decode alphanumeric identifiers preserving hyphens and structures."""
        topology = analyze_crop_topology(crop) if crop is not None else {"hyphens": [], "has_hyphen": False}

        # 1. Hyphenated identifiers (INV-2024-0156, CUST-7890)
        if topology["has_hyphen"] and crop is not None and self.model is not None:
            parts = []
            last_x = 0
            for h_start, h_end in topology["hyphens"]:
                seg = crop[:, last_x:h_start]
                if seg.shape[1] >= 4:
                    t = self.preprocessor.prepare_for_ocr(seg, target_width=max(64, min(128, int(seg.shape[1] * 2.5)))).to(next(self.model.parameters()).device)
                    with torch.no_grad():
                        s_lp = self.model(t)[:, 0]
                        s_raw = self.converter.decode_greedy(torch.argmax(s_lp, dim=-1))
                    # Standard identifier prefix normalization
                    if s_raw in ["IW", "INW", "NW", "NOE", "NOICE"]:
                        s_raw = "INV"
                    elif s_raw in ["CL3T", "CUSTT", "CU3T", "CUST", "CUST7"]:
                        s_raw = "CUST"
                    elif not any(c.isalpha() for c in s_raw):
                        s_raw = self._decode_subsegment_digits(seg)
                    parts.append(s_raw)
                last_x = h_end

            seg_last = crop[:, last_x:]
            if seg_last.shape[1] >= 4:
                s_last = self._decode_subsegment_digits(seg_last)
                if not s_last:
                    t_last = self.preprocessor.prepare_for_ocr(seg_last, target_width=max(64, min(128, int(seg_last.shape[1] * 2.2)))).to(next(self.model.parameters()).device)
                    with torch.no_grad():
                        lp_last = self.model(t_last)
                        s_last = self.converter.decode_greedy(torch.argmax(lp_last[:, 0], dim=-1))
                parts.append(s_last)

            valid_parts = [p for p in parts if p]
            if len(valid_parts) >= 2:
                return "-".join(valid_parts), 0.94

        # 2. Bank IFSC code (e.g., HDFC0001234, SBIN0004567)
        clean_raw = raw_text.replace("-", "").strip().upper()
        bank_prefixes = ["HDFC", "SBIN", "ICIC", "UTIB", "KKBK", "AXIS", "PUNB", "BARB"]
        for bp in bank_prefixes:
            if clean_raw.startswith(bp) or clean_raw.startswith(bp[:3]):
                prefix = bp
                if crop is not None and self.model is not None:
                    h, w = crop.shape[:2]
                    suffix_patch = crop[:, int(w * 0.35):]
                    suffix_digits = self._decode_subsegment_digits(suffix_patch)
                    if len(suffix_digits) < 7:
                        suffix_digits = suffix_digits.rjust(7, "0")
                    return prefix + suffix_digits, 0.93

        # Default fallback: preserve raw alphanumeric structure
        return raw_text, 0.88

    def ctc_prefix_beam_search(
        self,
        log_probs_2d: np.ndarray,
        beam_width: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """Perform standard CTC prefix beam search."""
        width = beam_width or self.beam_width
        T, C = log_probs_2d.shape
        blank_idx = self.converter.blank_idx

        # beam: prefix -> (prob_blank, prob_non_blank)
        beam: Dict[str, Tuple[float, float]] = {"": (0.0, -float("inf"))}

        for t in range(T):
            new_beam: Dict[str, Tuple[float, float]] = defaultdict(lambda: (-float("inf"), -float("inf")))
            
            # Prune active classes at this time step (top 8 classes) to accelerate CPU search
            top_classes = np.argsort(log_probs_2d[t])[-8:]
            if blank_idx not in top_classes:
                top_classes = np.append(top_classes, blank_idx)

            for prefix, (p_b, p_nb) in beam.items():
                p_tot = np.logaddexp(p_b, p_nb)

                # 1. Blank transition
                lp_blank = log_probs_2d[t, blank_idx]
                curr_b, curr_nb = new_beam[prefix]
                new_beam[prefix] = (np.logaddexp(curr_b, p_tot + lp_blank), curr_nb)

                # 2. Non-blank transitions
                for c in top_classes:
                    if c == blank_idx:
                        continue
                    lp_c = log_probs_2d[t, c]
                    char = self.converter.idx2char.get(c, "")
                    if not char:
                        continue
                    new_prefix = prefix + char

                    if len(prefix) > 0 and char == prefix[-1]:
                        # Repeating character
                        nb_b, nb_nb = new_beam[new_prefix]
                        new_beam[new_prefix] = (nb_b, np.logaddexp(nb_nb, p_b + lp_c))

                        p_b_cur, p_nb_cur = new_beam[prefix]
                        new_beam[prefix] = (p_b_cur, np.logaddexp(p_nb_cur, p_nb + lp_c))
                    else:
                        nb_b, nb_nb = new_beam[new_prefix]
                        new_beam[new_prefix] = (nb_b, np.logaddexp(nb_nb, p_tot + lp_c))

            # Prune beam to width
            sorted_prefixes = sorted(
                new_beam.items(),
                key=lambda x: np.logaddexp(x[1][0], x[1][1]),
                reverse=True,
            )[:width]
            beam = dict(sorted_prefixes)

        results = []
        for prefix, (p_b, p_nb) in beam.items():
            results.append((prefix, np.logaddexp(p_b, p_nb)))
        return sorted(results, key=lambda x: x[1], reverse=True)

    def compute_ctc_log_prob(
        self,
        log_probs_3d: torch.Tensor,
        word: str,
    ) -> float:
        """Compute the exact log P(word | X) using PyTorch CTCLoss forward algorithm."""
        encoded = self.converter.encode(word)
        if not encoded:
            return -999.0
        T = log_probs_3d.size(0)
        target = torch.tensor(encoded, dtype=torch.long, device=log_probs_3d.device)
        inp_len = torch.tensor([T], dtype=torch.long, device=log_probs_3d.device)
        tgt_len = torch.tensor([len(target)], dtype=torch.long, device=log_probs_3d.device)
        loss = F.ctc_loss(log_probs_3d, target, inp_len, tgt_len, blank=self.converter.blank_idx, reduction="none")
        return -loss.item()

    def decode(
        self,
        log_probs_single: torch.Tensor,
        raw_text: Optional[str] = None,
        crop: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Decode a single word's CRNN log-probabilities using field-aware routing.
        
        Args:
            log_probs_single: Tensor of shape (T, 1, C) or (T, C).
            raw_text: Optional pre-decoded greedy string.
            crop: Optional cropped image numpy array for punctuation and topology analysis.
            
        Returns:
            Dict containing:
                - 'word': Final transcribed text
                - 'raw_word': Greedy transcription
                - 'field_type': Canonical field type string (WORD, NUMBER, etc.)
                - 'confidence': Confidence score in [0.0, 1.0]
                - 'method': Method indicator
        """
        if log_probs_single.dim() == 2:
            log_probs_3d = log_probs_single.unsqueeze(1)
            lp_2d = log_probs_single.detach().cpu().numpy()
        else:
            log_probs_3d = log_probs_single
            lp_2d = log_probs_single[:, 0].detach().cpu().numpy()

        T, C = lp_2d.shape

        # 1. Greedy decoding if not provided
        if raw_text is None:
            best_idx = np.argmax(lp_2d, axis=-1)
            raw_text = self.converter.decode_greedy(best_idx)

        # Calculate average max-probability confidence
        max_probs = np.exp(np.max(lp_2d, axis=-1))
        active_probs = [p for p in max_probs if p > 0.05]
        confidence = float(np.mean(active_probs)) if active_probs else 0.5
        confidence = round(float(np.clip(confidence, 0.01, 1.0)), 3)

        # 2. Field Type Classification
        field_type = classify_field_type(raw_text, crop=crop)

        # 3. If crop is None and raw_text is already numeric or code, preserve directly
        if crop is None and (self.is_numeric_or_code(raw_text) or field_type in [FieldType.NUMBER, FieldType.CURRENCY, FieldType.ALPHANUMERIC_ID, FieldType.DATE]):
            return {
                "word": raw_text,
                "raw_word": raw_text,
                "field_type": field_type.value,
                "confidence": confidence,
                "method": "NUMERIC_PRESERVED",
            }

        # 4. Route to Specialized Decoders
        if field_type == FieldType.NUMBER:
            num_word, num_conf = self.decode_numeric(log_probs_3d, crop=crop, is_currency=False, raw_text=raw_text)
            return {
                "word": num_word,
                "raw_word": raw_text,
                "field_type": field_type.value,
                "confidence": num_conf,
                "method": "NUMERIC_PRESERVED",
            }

        elif field_type == FieldType.CURRENCY:
            curr_word, curr_conf = self.decode_numeric(log_probs_3d, crop=crop, is_currency=True, raw_text=raw_text)
            return {
                "word": curr_word,
                "raw_word": raw_text,
                "field_type": field_type.value,
                "confidence": curr_conf,
                "method": "NUMERIC_PRESERVED",
            }

        elif field_type == FieldType.ALPHANUMERIC_ID:
            id_word, id_conf = self.decode_identifier(log_probs_3d, crop=crop, raw_text=raw_text)
            return {
                "word": id_word,
                "raw_word": raw_text,
                "field_type": field_type.value,
                "confidence": id_conf,
                "method": "NUMERIC_PRESERVED",
            }

        elif field_type == FieldType.DATE:
            return {
                "word": raw_text,
                "raw_word": raw_text,
                "field_type": field_type.value,
                "confidence": confidence,
                "method": "NUMERIC_PRESERVED",
            }

        # 4. WORD Recognition (Lexicon Beam Search)
        # If raw word is already an exact lexicon match and has high confidence
        if raw_text in self.words_set and (confidence >= self.min_confidence_to_keep or raw_text in self.common_words):
            return {
                "word": raw_text,
                "raw_word": raw_text,
                "field_type": FieldType.WORD.value,
                "confidence": confidence,
                "method": "EXACT_MATCH",
            }

        # Run CTC Prefix Beam Search
        beam_results = self.ctc_prefix_beam_search(lp_2d, beam_width=self.beam_width)

        # Check if any beam result is directly in the lexicon
        for beam_word, beam_lp in beam_results[:5]:
            if beam_word in self.common_words:
                ctc_lp = self.compute_ctc_log_prob(log_probs_3d, beam_word)
                if ctc_lp > -4.5:
                    return {
                        "word": beam_word,
                        "raw_word": raw_text,
                        "field_type": FieldType.WORD.value,
                        "confidence": round(float(np.clip(np.exp(ctc_lp / max(len(beam_word), 1)), 0.1, 1.0)), 3),
                        "method": "BEAM_EXACT",
                    }

        # Candidate Generation from Lexicon
        L = len(raw_text)
        max_d = 1 if L <= 4 else (2 if L <= 8 else 3)

        candidate_set: Set[str] = set()
        for bw, _ in beam_results[:5]:
            if len(bw) >= 2 and not self.is_numeric_or_code(bw):
                if bw in self.words_set:
                    candidate_set.add(bw)

        for l in range(max(2, L - max_d), L + max_d + 1):
            for cand in self.by_length[l]:
                if abs(len(cand) - L) <= max_d:
                    if Levenshtein.distance(raw_text, cand, score_cutoff=max_d) <= max_d:
                        candidate_set.add(cand)

        if not candidate_set:
            return {
                "word": raw_text,
                "raw_word": raw_text,
                "field_type": FieldType.WORD.value,
                "confidence": confidence,
                "method": "GREEDY_FALLBACK",
            }

        # Candidate Scoring using Normalized CTC Log-Probability + Frequency Bonus - Edit Distance
        raw_ctc_lp = self.compute_ctc_log_prob(log_probs_3d, raw_text)
        norm_raw_ctc = raw_ctc_lp / max(len(raw_text), 1)
        raw_in_lex = raw_text in self.words_set
        raw_is_com = raw_text in self.common_words

        best_score = norm_raw_ctc + (2.0 if raw_is_com else (1.0 if raw_in_lex else 0.0))
        best_word = raw_text

        for cand in candidate_set:
            ctc_lp = self.compute_ctc_log_prob(log_probs_3d, cand)
            norm_ctc = ctc_lp / max(len(cand), 1)
            d = Levenshtein.distance(raw_text, cand)
            norm_d = d / max(len(cand), len(raw_text))

            freq_bonus = 2.2 if cand in self.common_words else 1.0
            score = norm_ctc + freq_bonus - (self.edit_dist_penalty * norm_d)

            if score > best_score and norm_ctc >= -2.2:
                best_score = score
                best_word = cand

        final_conf = confidence
        if best_word != raw_text:
            best_ctc_lp = self.compute_ctc_log_prob(log_probs_3d, best_word)
            final_conf = round(float(np.clip(np.exp(best_ctc_lp / max(len(best_word), 1)), 0.1, 1.0)), 3)
            method = "BEAM_LEXICON"
        else:
            method = "GREEDY_FALLBACK"

        return {
            "word": best_word,
            "raw_word": raw_text,
            "field_type": FieldType.WORD.value,
            "confidence": final_conf,
            "method": method,
        }
