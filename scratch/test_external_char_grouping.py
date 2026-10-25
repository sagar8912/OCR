"""Test character-to-word grouping on RETR_EXTERNAL contours."""

import cv2
import numpy as np

img = cv2.imread("sample/invoice_test.png")
h_img, w_img = img.shape[:2]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
grad = cv2.magnitude(grad_x, grad_y)
grad = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
_, binary = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

h_kernel_len = max(45, int(w_img * 0.045))
v_kernel_len = max(35, int(h_img * 0.055))
h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_len, 1)))
v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_len)))
lines = cv2.dilate(cv2.bitwise_or(h_lines, v_lines), np.ones((3, 3), np.uint8), iterations=2)
text_mask = cv2.bitwise_and(binary, cv2.bitwise_not(lines))

cnts_ext, _ = cv2.findContours(text_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"1. Raw RETR_EXTERNAL contours: {len(cnts_ext)}")

# Filter noise and borders
filtered = []
for cnt in cnts_ext:
    x, y, w, h = cv2.boundingRect(cnt)
    area = w * h
    if area < 15 or w < 2 or h < 5:
        continue
    if w > 0.85 * w_img or h > 0.35 * h_img:
        continue
    if y < 3 and h > 0.4 * h_img:
        continue
    if x < 3 and w > 0.4 * w_img:
        continue
    filtered.append([y, x, y + h, x + w])

print(f"2. Filtered characters: {len(filtered)}")

# Now group characters into words:
# Two components C1 and C2 belong to the SAME WORD if:
# 1. Height similarity: min_h / max_h >= 0.4
# 2. Vertical overlap: overlap_y / min_h >= 0.45 or abs(yc1 - yc2) <= 0.45 * max_h
# 3. Horizontal proximity: gap dx >= -3 and dx <= max(4, int(min_h * 0.42))
words = [[t, l, b, r] for t, l, b, r in filtered]

changed = True
while changed:
    changed = False
    words = sorted(words, key=lambda b: (b[1], b[0]))
    new_words = []
    skip = set()

    for i in range(len(words)):
        if i in skip:
            continue
        c_top, c_left, c_bot, c_right = words[i]
        c_h = c_bot - c_top
        c_yc = (c_top + c_bot) / 2.0

        for j in range(i + 1, len(words)):
            if j in skip:
                continue
            o_top, o_left, o_bot, o_right = words[j]
            o_h = o_bot - o_top
            o_yc = (o_top + o_bot) / 2.0

            min_h = min(c_h, o_h)
            max_h = max(c_h, o_h)
            if min_h == 0 or (min_h / max_h) < 0.4:
                continue

            y_overlap = max(0, min(c_bot, o_bot) - max(c_top, o_top))
            if (y_overlap / min_h) < 0.40 and abs(c_yc - o_yc) > 0.45 * max_h:
                continue

            dx = o_left - c_right
            max_gap = max(4, int(min_h * 0.42))

            if -4 <= dx <= max_gap:
                c_top = min(c_top, o_top)
                c_left = min(c_left, o_left)
                c_bot = max(c_bot, o_bot)
                c_right = max(c_right, o_right)
                c_h = c_bot - c_top
                c_yc = (c_top + c_bot) / 2.0
                skip.add(j)
                changed = True

        new_words.append([c_top, c_left, c_bot, c_right])
    words = new_words

print(f"3. Grouped words: {len(words)}")

# Filter any tiny isolated artifacts (<35 px area)
final_words = [b for b in words if (b[2]-b[0])*(b[3]-b[1]) >= 35]
print(f"4. Final words: {len(final_words)}")

# Sort reading order
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

print("\n--- Inspecting First 25 Words ---")
from src.engine.recognizer import TextRecognizer
recognizer = TextRecognizer(model_path="models/best_model.pt", device="cpu")

for idx, (t, l, b, r) in enumerate(sorted_boxes[:25]):
    crop = img[t:b, l:r]
    w = r - l
    h = b - t
    pred = recognizer.recognize_crop(crop)
    print(f"Word {idx:2d} [{t:3d}, {l:3d}, {b:3d}, {r:3d}] ({w:3d}x{h:2d}): Pred = '{pred}'")
