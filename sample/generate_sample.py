"""Utility script to generate sample multi-word document images for testing."""

import cv2
import numpy as np


def generate_document_sample(output_path: str = "sample/document_sample.png"):
    """Generate a clean synthetic financial document snippet with multiple words."""
    h, w = 300, 600
    # Light clean document background
    canvas = np.full((h, w, 3), 245, dtype=np.uint8)

    # Add header banner line
    cv2.line(canvas, (30, 70), (570, 70), (200, 200, 200), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    # Line 1: Header
    cv2.putText(canvas, "INVOICE", (40, 50), font, 1.2, (20, 20, 20), 3, cv2.LINE_AA)
    cv2.putText(canvas, "CONFIDENTIAL", (360, 50), font, 0.7, (50, 50, 180), 2, cv2.LINE_AA)

    # Line 2: Account info
    cv2.putText(canvas, "ACCOUNT", (40, 120), font, 0.8, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.putText(canvas, "987654", (220, 120), font, 0.8, (20, 20, 20), 2, cv2.LINE_AA)

    # Line 3: Amount
    cv2.putText(canvas, "BALANCE", (40, 180), font, 0.8, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.putText(canvas, "50000", (220, 180), font, 0.8, (20, 20, 20), 2, cv2.LINE_AA)

    # Line 4: Status
    cv2.putText(canvas, "STATUS", (40, 240), font, 0.8, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.putText(canvas, "VERIFIED", (220, 240), font, 0.8, (20, 120, 20), 2, cv2.LINE_AA)

    cv2.imwrite(output_path, canvas)
    print(f"Sample generated: {output_path}")


if __name__ == "__main__":
    generate_document_sample()
