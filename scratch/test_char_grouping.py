"""Test bottom-up Character-to-Word Grouper on invoice_test.png."""

import os
import cv2
import numpy as np

def run_char_to_word_grouping():
    img = cv2.imread("sample/invoice_test.png")
    h_img, w_img = img.shape[:2]
    print(f"Loaded invoice: {w_img}x{h_img}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Edge/gradient magnitude to detect character strokes
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(grad_x, grad_y)
    grad = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Otsu threshold
    _, binary = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 2. Extract and remove long lines (table borders and page border)
    # Horizontal line kernel (at least 45px or 4.5% of width)
    h_kernel_len = max(45, int(w_img * 0.045))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_len, 1)))

    # Vertical line kernel (at least 35px or 5% of height)
    v_kernel_len = max(35, int(h_img * 0.05))
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_len)))

    lines = cv2.bitwise_or(h_lines, v_lines)
    lines_dilated = cv2.dilate(lines, np.ones((3, 3), np.uint8), iterations=2)
    text_mask = cv2.bitwise_and(binary, cv2.bitwise_not(lines_dilated))

    # 3. Connect close character strokes (intra-character, e.g. crossbars of T, H, dots of i)
    char_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    stroke_connected = cv2.morphologyEx(text_mask, cv2.MORPH_CLOSE, char_kernel)

    # 4. Connected components / Contours to get character candidates
    contours, _ = cv2.findContours(stroke_connected, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Raw contours found: {len(contours)}")

    raw_chars = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        # Filter noise
        if area < 20 or w < 3 or h < 6:
            continue
        # Filter outer borders
        if w > 0.80 * w_img or h > 0.35 * h_img:
            continue
        if y < 3 and h > 0.4 * h_img:
            continue
        if x < 3 and w > 0.4 * w_img:
            continue
        # Filter thin lines
        if (w / max(1, h) > 20) or (h / max(1, w) > 15):
            continue
        raw_chars.append([y, x, y + h, x + w])

    print(f"Filtered character components: {len(raw_chars)}")

    # 5. Bottom-up Character-to-Word Grouping
    # Sort characters top-to-bottom, left-to-right
    # Concept: Characters C_a and C_b belong to the SAME WORD if:
    # 1. Vertical alignment: y_overlap >= 0.4 * min(h_a, h_b) and abs(center_y_a - center_y_b) <= 0.4 * max_h
    # 2. Height ratio: min_h / max_h >= 0.45
    # 3. Horizontal proximity: gap dx >= -2 (overlap/kerning) and dx <= max(4, int(min_h * 0.40))
    # This strictly prevents bridging across inter-word spaces (which are >= 0.50 * h)!

    # Group into words
    # Start with each character as an independent bounding box
    words = [[t, l, b, r] for t, l, b, r in raw_chars]
    
    changed = True
    iteration = 0
    while changed and iteration < 15:
        changed = False
        iteration += 1
        words = sorted(words, key=lambda b: (b[1], b[0]))
        new_words = []
        skip_indices = set()

        for i in range(len(words)):
            if i in skip_indices:
                continue
            curr = words[i]
            c_top, c_left, c_bottom, c_right = curr
            c_h = c_bottom - c_top
            c_yc = (c_top + c_bottom) / 2.0

            for j in range(i + 1, len(words)):
                if j in skip_indices:
                    continue
                other = words[j]
                o_top, o_left, o_bottom, o_right = other
                o_h = o_bottom - o_top
                o_yc = (o_top + o_bottom) / 2.0

                # Check vertical alignment
                y_overlap = max(0, min(c_bottom, o_bottom) - max(c_top, o_top))
                min_h = min(c_h, o_h)
                max_h = max(c_h, o_h)

                if min_h == 0 or (min_h / max_h) < 0.45:
                    continue
                if (y_overlap / min_h) < 0.40 and abs(c_yc - o_yc) > 0.40 * max_h:
                    continue

                # Check horizontal gap
                # Other is to the right of curr or overlapping
                dx = o_left - c_right
                max_intra_word_gap = max(4, int(min_h * 0.38))

                if -4 <= dx <= max_intra_word_gap:
                    # Merge into curr!
                    c_top = min(c_top, o_top)
                    c_left = min(c_left, o_left)
                    c_bottom = max(c_bottom, o_bottom)
                    c_right = max(c_right, o_right)
                    curr = [c_top, c_left, c_bottom, c_right]
                    c_h = c_bottom - c_top
                    c_yc = (c_top + c_bottom) / 2.0
                    skip_indices.add(j)
                    changed = True

            new_words.append(curr)

        words = new_words

    print(f"Grouped words count: {len(words)}")

    # Filter out any tiny remaining standalone punctuation/dots that didn't merge
    final_words = []
    for b in words:
        t, l, bot, r = b
        w = r - l
        h = bot - t
        if w * h < 50 and (w < 8 or h < 8):
            continue
        final_words.append(b)

    print(f"Final valid words count: {len(final_words)}")

    # Sort in reading order
    boxes_with_centers = [((b[0] + b[2]) / 2.0, b[2] - b[0], b) for b in final_words]
    lines = []
    for yc, bh, b in sorted(boxes_with_centers, key=lambda x: x[0]):
        placed = False
        for line in lines:
            if abs(yc - line["yc"]) < max(bh, line["h"]) * 0.45:
                line["boxes"].append(b)
                placed = True
                break
        if not placed:
            lines.append({"yc": yc, "h": bh, "boxes": [b]})

    sorted_boxes = []
    for line in sorted(lines, key=lambda l: l["yc"]):
        line_boxes = sorted(line["boxes"], key=lambda b: b[1])
        sorted_boxes.extend(line_boxes)

    # Let's inspect some of the grouped words:
    # Check if "INVOICE", "Total", "Amount", "Laptop", "Dell", "Inspiron", "CONFIDENTIAL" are words
    vis = img.copy()
    for idx, (t, l, b, r) in enumerate(sorted_boxes):
        cv2.rectangle(vis, (l, t), (r, b), (0, 255, 0), 1)
        cv2.putText(vis, str(idx), (l, max(12, t - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    
    os.makedirs("sample/diagnostic_char_grouping", exist_ok=True)
    cv2.imwrite("sample/diagnostic_char_grouping/final_words.png", vis)
    print("Saved sample/diagnostic_char_grouping/final_words.png")

    # Test OCR recognizer on these clean words
    from src.engine.recognizer import TextRecognizer
    recognizer = TextRecognizer(model_path="models/best_model.pt", device="cpu")

    print("\n--- Sample Recognized Words ---")
    for idx, (t, l, b, r) in enumerate(sorted_boxes[:35]):
        w = r - l
        h = b - t
        crop = img[t:b, l:r]
        pred = recognizer.recognize_crop(crop)
        print(f"Word {idx:2d} [{t:3d}, {l:3d}, {b:3d}, {r:3d}] ({w:3d}x{h:2d}): Pred = '{pred}'")

if __name__ == "__main__":
    run_char_to_word_grouping()
