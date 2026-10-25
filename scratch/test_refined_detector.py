"""Prototype of refined detector with adaptive grouping and vertical projection word splitting."""

import os
import cv2
import numpy as np

def run_refined_detector(image_path: str, debug: bool = True):
    img = cv2.imread(image_path)
    h_img, w_img = img.shape[:2]
    print(f"Processing {image_path}: {w_img}x{h_img}")

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # If single-word crop
    if h_img <= 80 and w_img <= 350:
        return [[0, 0, h_img, w_img]]

    # 1. Edge/gradient magnitude
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(grad_x, grad_y)
    grad = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 2. Otsu thresholding
    _, binary = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. Detect and remove horizontal and vertical lines (table rules, underlines, page frames)
    h_kernel_len = max(25, int(w_img * 0.035))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_len, 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)

    v_kernel_len = max(20, int(h_img * 0.035))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_len))
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

    lines_mask = cv2.bitwise_or(h_lines, v_lines)
    # Dilate lines slightly to eliminate border artifacts
    lines_dilated = cv2.dilate(lines_mask, np.ones((3, 3), np.uint8), iterations=2)
    text_mask = cv2.bitwise_and(binary, cv2.bitwise_not(lines_dilated))

    # 4. Connect characters into text elements
    # Small morphological closing to connect character strokes
    kw = max(4, int(w_img * 0.007))  # ~7px
    kh = max(2, int(h_img * 0.004))  # ~2-3px
    word_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
    connected = cv2.morphologyEx(text_mask, cv2.MORPH_CLOSE, word_kernel)

    # 5. Contours
    raw_contours, _ = cv2.findContours(connected, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    num_raw = len(raw_contours)

    # Filter noise, page borders, and line artifacts
    filtered_boxes = []
    for cnt in raw_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        # Ignore tiny noise (<45 px^2, <8px wide, <7px tall)
        if area < 45 or w < 8 or h < 7:
            continue
        # Ignore outer document border
        if w > 0.85 * w_img or h > 0.35 * h_img:
            continue
        if y < 3 and h > 0.4 * h_img:
            continue
        if x < 3 and w > 0.4 * w_img:
            continue
        # Ignore extreme line-like artifacts
        if h <= 2 or w <= 2:
            continue
        if (w / h > 25) or (h / w > 15):
            continue

        filtered_boxes.append([y, x, y + h, x + w])

    num_filtered = len(filtered_boxes)

    # 6. Height-adaptive merging of intra-word fragments on same baseline
    # Sort left-to-right
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
                # Dynamic horizontal gap tolerance based on font height
                # For large font (h=35, INVOICE), allow gap up to 18px. For small (h=14), allow up to 7px.
                gap_allowed = max(5, int(min_h * 0.48))
                if abs(o_left - c_right) <= gap_allowed or (o_left <= c_right and o_right >= c_left):
                    new_top = min(c_top, o_top)
                    new_left = min(c_left, o_left)
                    new_bottom = max(c_bottom, o_bottom)
                    new_right = max(c_right, o_right)
                    curr = [new_top, new_left, new_bottom, new_right]
                    sorted_by_left.pop(i)
                    sorted_by_left.insert(0, curr)
                    merged_any = True
                    break
        if not merged_any:
            merged.append(curr)

    num_merged = len(merged)

    # 7. Word Segmentation via Vertical Projection Profile
    # Any box that is significantly wide (aspect ratio > 2.5 and width > 60)
    # may contain multiple words separated by inter-word spaces.
    final_boxes = []
    for box in merged:
        t, l, b, r = box
        w = r - l
        h = b - t
        aspect = w / max(1, h)

        if aspect > 2.2 and w > 45:
            # Check 1D vertical projection profile inside text_mask
            sub = text_mask[t:b, l:r]
            col_proj = np.sum(sub > 0, axis=0)

            # An inter-word space is a valley of zeros or near-zeros
            min_space_width = max(3, int(h * 0.22))
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
                            if (end_x - start_x) >= 6:
                                words_in_box.append([t, l + start_x, b, l + end_x])
                            in_word = False
                            zero_count = 0

            if in_word:
                end_x = w
                if (end_x - start_x) >= 6:
                    words_in_box.append([t, l + start_x, b, l + end_x])

            if len(words_in_box) > 1:
                final_boxes.extend(words_in_box)
            else:
                final_boxes.append(box)
        else:
            final_boxes.append(box)

    # 8. Sort in reading order
    boxes_with_centers = []
    for b in final_boxes:
        y_center = (b[0] + b[2]) / 2.0
        height = b[2] - b[0]
        boxes_with_centers.append((y_center, height, b))

    lines = []
    for yc, bh, b in sorted(boxes_with_centers, key=lambda x: x[0]):
        placed = False
        for line in lines:
            if abs(yc - line["yc"]) < max(bh, line["h"]) * 0.5:
                line["boxes"].append(b)
                placed = True
                break
        if not placed:
            lines.append({"yc": yc, "h": bh, "boxes": [b]})

    sorted_boxes = []
    for line in sorted(lines, key=lambda l: l["yc"]):
        line_boxes = sorted(line["boxes"], key=lambda b: b[1])
        sorted_boxes.extend(line_boxes)

    print(f"Stage 1 Report:")
    print(f"  Raw contours/components: {num_raw}")
    print(f"  After noise & border filtering: {num_filtered}")
    print(f"  After baseline/gap grouping: {num_merged}")
    print(f"  Final text regions (after projection word splitting): {len(sorted_boxes)}")

    if debug:
        os.makedirs("sample/debug_detector_v2", exist_ok=True)
        cv2.imwrite("sample/debug_detector_v2/1_grayscale.png", gray)
        cv2.imwrite("sample/debug_detector_v2/2_gradient.png", grad)
        cv2.imwrite("sample/debug_detector_v2/3_binary.png", binary)
        cv2.imwrite("sample/debug_detector_v2/4_lines_mask.png", lines_mask)
        cv2.imwrite("sample/debug_detector_v2/5_text_mask.png", text_mask)
        cv2.imwrite("sample/debug_detector_v2/6_connected.png", connected)

        vis = img.copy()
        for idx, (top, left, bottom, right) in enumerate(sorted_boxes):
            cv2.rectangle(vis, (left, top), (right, bottom), (0, 255, 0), 1)
            cv2.putText(vis, str(idx), (left, max(12, top - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        cv2.imwrite("sample/debug_detector_v2/7_final_boxes.png", vis)
        print("Debug images saved to sample/debug_detector_v2/")

    return sorted_boxes

if __name__ == "__main__":
    boxes = run_refined_detector("sample/invoice_test.png", debug=True)
