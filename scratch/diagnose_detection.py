"""Diagnostic script for Phase 1: Detection investigation."""

import os
import cv2
import numpy as np

def run_detection_diagnostic():
    img_path = "sample/invoice_test.png"
    img = cv2.imread(img_path)
    h_img, w_img = img.shape[:2]
    print(f"Loaded {img_path}: {w_img}x{h_img}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Edge/gradient magnitude
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(grad_x, grad_y)
    grad = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 2. Otsu thresholding
    _, binary = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. Horizontal & Vertical lines
    h_kernel_len = max(25, int(w_img * 0.04))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_len, 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)

    v_kernel_len = max(20, int(h_img * 0.04))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_len))
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

    lines_mask = cv2.bitwise_or(h_lines, v_lines)
    lines_dilated = cv2.dilate(lines_mask, np.ones((3, 3), np.uint8), iterations=1)
    text_mask = cv2.bitwise_and(binary, cv2.bitwise_not(lines_dilated))

    # Current word closing kernel:
    kw = min(12, max(6, int(w_img * 0.01))) # 10 px
    kh = max(2, int(h_img * 0.005)) # 3 px
    word_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
    connected = cv2.morphologyEx(text_mask, cv2.MORPH_CLOSE, word_kernel)

    # Contours
    contours, _ = cv2.findContours(connected, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Total raw contours found: {len(contours)}")

    os.makedirs("sample/diagnostic_phase1", exist_ok=True)
    cv2.imwrite("sample/diagnostic_phase1/gray.png", gray)
    cv2.imwrite("sample/diagnostic_phase1/grad.png", grad)
    cv2.imwrite("sample/diagnostic_phase1/binary.png", binary)
    cv2.imwrite("sample/diagnostic_phase1/lines_mask.png", lines_mask)
    cv2.imwrite("sample/diagnostic_phase1/text_mask.png", text_mask)
    cv2.imwrite("sample/diagnostic_phase1/connected.png", connected)

    # Now inspect all contours and classify them
    raw_boxes = []
    tiny_boxes = []
    border_boxes = []
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < 35 or w < 8 or h < 7:
            tiny_boxes.append((x, y, w, h))
            continue
        if w > 0.85 * w_img or h > 0.35 * h_img:
            border_boxes.append((x, y, w, h))
            continue
        if y < 2 and h > 0.5 * h_img:
            border_boxes.append((x, y, w, h))
            continue
        raw_boxes.append([y, x, y + h, x + w])

    print(f"Filtered raw boxes: {len(raw_boxes)} (tiny: {len(tiny_boxes)}, border: {len(border_boxes)})")

    # Let's see what each box is!
    # Let's inspect the boxes around "INVOICE", "ABC TECHNOLOGIES", etc.
    from src.engine.detector import TextDetector
    detector = TextDetector()
    final_boxes = detector.detect(img)
    print(f"Final detector boxes after merge & sort: {len(final_boxes)}")

    # Draw final boxes on image with box indices
    debug_vis = img.copy()
    for idx, (t, l, b, r) in enumerate(final_boxes):
        w = r - l
        h = b - t
        cv2.rectangle(debug_vis, (l, t), (r, b), (0, 255, 0), 1)
        cv2.putText(debug_vis, str(idx), (l, max(12, t - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    cv2.imwrite("sample/diagnostic_phase1/final_boxes_annotated.png", debug_vis)

    # Print first 25 boxes and their sizes
    print("\nSample of detected boxes:")
    for idx, (t, l, b, r) in enumerate(final_boxes[:25]):
        w = r - l
        h = b - t
        aspect = w / max(1, h)
        print(f"Box {idx:2d}: [t={t:3d}, l={l:3d}, b={b:3d}, r={r:3d}] (w={w:3d}, h={h:2d}, aspect={aspect:.2f})")

if __name__ == "__main__":
    run_detection_diagnostic()
