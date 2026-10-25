"""Test grouping parameters to ensure whole words are formed without character fragmentation."""

import os
import cv2
import numpy as np

def run_grouping_test(min_space_ratio=0.45, kw_val=10):
    img = cv2.imread("sample/invoice_test.png")
    h_img, w_img = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Edge/gradient
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(grad_x, grad_y)
    grad = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, binary = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 2. Line removal
    h_kernel_len = max(25, int(w_img * 0.035))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_len, 1)))
    v_kernel_len = max(20, int(h_img * 0.035))
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_len)))
    lines = cv2.dilate(cv2.bitwise_or(h_lines, v_lines), np.ones((3, 3), np.uint8), iterations=2)
    text_mask = cv2.bitwise_and(binary, cv2.bitwise_not(lines))

    # 3. Morphological closing to connect characters into words
    word_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw_val, 3))
    connected = cv2.morphologyEx(text_mask, cv2.MORPH_CLOSE, word_kernel)

    raw_contours, _ = cv2.findContours(connected, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter
    filtered_boxes = []
    for cnt in raw_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if (w * h < 45) or w < 8 or h < 7:
            continue
        if w > 0.85 * w_img or h > 0.35 * h_img:
            continue
        if y < 3 and h > 0.4 * h_img:
            continue
        if x < 3 and w > 0.4 * w_img:
            continue
        if (w / h > 25) or (h / w > 15):
            continue
        filtered_boxes.append([y, x, y + h, x + w])

    # Dynamic baseline merging
    sorted_by_left = sorted(filtered_boxes, key=lambda b: (b[1], b[0]))
    merged = []
    while sorted_by_left:
        curr = sorted_by_left.pop(0)
        c_top, c_left, c_bottom, c_right = curr
        c_h = c_bottom - c_top

        merged_any = False
        for i, other in enumerate(sorted_by_left):
            o_top, o_left, o_bottom, o_right = other
            o_h = o_bottom - o_top
            y_overlap = max(0, min(c_bottom, o_bottom) - max(c_top, o_top))
            min_h = min(c_h, o_h)

            if min_h > 0 and (y_overlap / min_h) >= 0.5:
                gap_allowed = max(5, int(min_h * 0.45))
                if abs(o_left - c_right) <= gap_allowed or (o_left <= c_right and o_right >= c_left):
                    curr = [min(c_top, o_top), min(c_left, o_left), max(c_bottom, o_bottom), max(c_right, o_right)]
                    sorted_by_left.pop(i)
                    sorted_by_left.insert(0, curr)
                    merged_any = True
                    break
        if not merged_any:
            merged.append(curr)

    # Word segmentation on wide boxes using vertical projection profile
    final_boxes = []
    for box in merged:
        t, l, b, r = box
        w = r - l
        h = b - t
        aspect = w / max(1, h)

        # Only split if aspect ratio is wide enough to be multiple words
        if aspect > 2.8 and w > 55:
            sub = text_mask[t:b, l:r]
            col_proj = np.sum(sub > 0, axis=0)
            min_space_width = max(5, int(h * min_space_ratio))

            words_in_box = []
            in_word = False
            start_x = 0
            zero_count = 0

            for x_idx in range(w):
                val = col_proj[x_idx]
                if val > 0:
                    if not in_word:
                        start_x = max(0, x_idx - 1)
                        in_word = True
                    zero_count = 0
                else:
                    if in_word:
                        zero_count += 1
                        if zero_count >= min_space_width:
                            end_x = x_idx - zero_count + 1
                            if (end_x - start_x) >= 8:
                                words_in_box.append([t, l + start_x, b, l + end_x])
                            in_word = False
                            zero_count = 0

            if in_word:
                end_x = w
                if (end_x - start_x) >= 8:
                    words_in_box.append([t, l + start_x, b, l + end_x])

            if len(words_in_box) > 1:
                final_boxes.extend(words_in_box)
            else:
                final_boxes.append(box)
        else:
            final_boxes.append(box)

    print(f"Ratio={min_space_ratio}, kw={kw_val}: Raw={len(raw_contours)}, Filtered={len(filtered_boxes)}, Merged={len(merged)}, Final={len(final_boxes)}")
    return final_boxes

if __name__ == "__main__":
    for ratio in [0.40, 0.45, 0.50]:
        for kw in [10, 12]:
            run_grouping_test(min_space_ratio=ratio, kw_val=kw)
