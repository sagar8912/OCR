"""Unit and regression tests for LexiconDecoder and CTC Beam Search."""

import numpy as np
import pytest
import torch

from src.engine.decoder import LexiconDecoder
from src.training.dataset import LabelConverter, DEFAULT_CONVERTER


@pytest.fixture(scope="module")
def decoder():
    return LexiconDecoder()


def test_numeric_preservation(decoder):
    """Numbers, dates, and account numbers must never be altered by lexicon lookup."""
    samples = [
        "12345",
        "24/09/2026",
        "150,450",
        "9876543210",
        "44",
        "2001",
        "0.99",
    ]
    for s in samples:
        assert decoder.is_numeric_or_code(s) is True
        res = decoder.decode(torch.randn(32, 1, 37), raw_text=s)
        assert res["word"] == s
        assert res["method"] == "NUMERIC_PRESERVED"


def test_alphanumeric_code_preservation(decoder):
    """Customer IDs, invoice numbers, and alphanumeric codes must not be converted to dictionary words."""
    codes = [
        "CUST-7890",
        "INV-2026-001",
        "HDFC001234",
        "A12B34",
        "INR",
        "USD",
    ]
    for c in codes:
        assert decoder.is_numeric_or_code(c) is True
        res = decoder.decode(torch.randn(32, 1, 37), raw_text=c)
        assert res["word"] == c
        assert res["method"] == "NUMERIC_PRESERVED"


def test_exact_lexicon_matching(decoder):
    """Exact lexicon words should be recognized as EXACT_MATCH or retained directly."""
    for word in ["TOTAL", "BANK", "DATE", "AMOUNT", "INVOICE"]:
        assert word in decoder.words_set
        # Create synthetic perfect logits for this word
        T = 32
        C = 37
        logits = torch.full((T, 1, C), -10.0)
        enc = DEFAULT_CONVERTER.encode(word)
        step = T // len(enc)
        for i, char_idx in enumerate(enc):
            logits[i * step, 0, char_idx] = 10.0
            # set blank between
            if i * step + 1 < T:
                logits[i * step + 1, 0, 0] = 5.0

        res = decoder.decode(logits, raw_text=word)
        assert res["word"] == word
        assert res["confidence"] >= 0.70


def test_ctc_prefix_beam_search(decoder):
    """CTC prefix beam search should return high-likelihood candidates."""
    T = 20
    C = 37
    lp = np.full((T, C), -20.0)
    # Blank (index 0) is high by default in CTC background
    lp[:, 0] = 0.0
    # Simulate character activations for 'CAT'
    enc = DEFAULT_CONVERTER.encode("CAT")  # [C, A, T]
    lp[2, 0] = -10.0
    lp[2, enc[0]] = 5.0
    lp[7, 0] = -10.0
    lp[7, enc[1]] = 5.0
    lp[12, 0] = -10.0
    lp[12, enc[2]] = 5.0

    beams = decoder.ctc_prefix_beam_search(lp, beam_width=5)
    assert len(beams) > 0
    words = [b[0] for b in beams]
    assert "CAT" in words


def test_unknown_word_fallback(decoder):
    """Unknown nonsense words with no close lexicon matches should gracefully fallback."""
    fake_word = "XZQWJK"
    assert fake_word not in decoder.words_set
    logits = torch.randn(32, 1, 37)
    res = decoder.decode(logits, raw_text=fake_word)
    # Must not crash and should return a valid string
    assert isinstance(res["word"], str)
    assert "confidence" in res
