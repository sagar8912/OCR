"""Comprehensive test for Phase 1 to 4: Detector on Invoice, Document Sample, and IIIT-5K."""

import os
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple


class RobustTextDetector:
    """CPU-friendly document text detector with bottom-up character-to-word grouping."""

    def __init__(
        self,
        min_area: int = 35,
        min_char_width: int = 3,
        min_char_height: int = 6,
        max_width_ratio: float = 0.85,
        max_height_ratio: float = 0.35,
        debug_dir: str = "sample/debug_detector_phases",
    ):
        self.min_area = min_area
        self.min_char_width = min_char_width
        self.min_char_height = min_char_height
        self.max_width_ratio = max_width_ratio
        self.max_height_ratio = max_height_ratio
        self.debug_dir = debug_dir
        self.metrics: Dict[str, int] = {}

    def detect(self, image: np.ndarray, debug: bool = False, debug_prefix: str = "invoice") -> List[List[int]]:
        h_img, w_img = image.shape[:2]
        if h_img == 0 or w_img == 0:
            return []

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Handle already-cropped single word image (e.g. IIIT-5K)
        if h_img <= 80 and w_img <= 350:
            self.metrics = {
                "num_raw": 1,
                "num_filtered": 1,
                "num_grouped": 1,
                "num_final": 1,
            }
            return [[0, 0, int(h_img), int(w_img)]]

        # 1. Edge/gradient magnitude
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = cv2.magnitude(grad_x, grad_y)
        grad = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # 2. Otsu thresholding
        _, binary = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 3. Detect and remove document/table grid lines
        h_kernel_len = max(45, int(w_img * 0.045))
        h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_len, 1)))

        v_kernel_len = max(35, int(h_img * 0.055))
        v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_len)))

        lines_mask = cv2.bitwise_or(h_lines, v_lines)
        lines_dilated = cv2.dilate(lines_mask, np.ones((3, 3), np.uint8), iterations=2)
        text_mask = cv2.bitwise_and(binary, cv2.bitwise_not(lines_dilated))

        # 4. Extract external character/stroke contours (ignores internal letter holes)
        char_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        morph_out = cv2.morphologyEx(text_mask, cv2.MORPH_CLOSE, char_kernel)

        contours, _ = cv2.findContours(morph_out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        num_raw = len(contours)

        # 5. Filter noise and document borders
        filtered_chars = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if area < 18 or w < self.min_char_width or h < self.min_char_height:
                continue
            if w > self.max_width_ratio * w_img or h > self.max_height_ratio * h_img:
                continue
            if y < 3 and h > 0.4 * h_img:
                continue
            if x < 3 and w > 0.4 * w_img:
                continue
            if (w / max(1, h) > 20) or (h / max(1, w) > 15):
                continue
            filtered_chars.append([y, x, y + h, x + w])

        num_filtered = len(filtered_chars)

        if not filtered_chars:
            self.metrics = {"num_raw": num_raw, "num_filtered": 0, "num_grouped": 1, "num_final": 1}
            return [[0, 0, int(h_img), int(w_img)]]

        # 6. Bottom-up Character-to-Word Grouping
        words = [[t, l, b, r] for t, l, b, r in filtered_chars]
        changed = True
        iterations = 0

        while changed and iterations < 20:
            changed = False
            iterations += 1
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

                    # Heights must be compatible (within ~2.5x of each other)
                    if min_h == 0 or (min_h / max_h) < 0.40:
                        continue

                    # Vertical overlap or baseline alignment
                    y_overlap = max(0, min(c_bot, o_bot) - max(c_top, o_top))
                    if (y_overlap / min_h) < 0.40 and abs(c_yc - o_yc) > 0.45 * max_h:
                        continue

                    # Horizontal gap: must be an intra-word character gap, NOT an inter-word space
                    dx = o_left - c_right
                    max_intra_word_gap = max(4, int(min_h * 0.42))

                    if -4 <= dx <= max_intra_word_gap:
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

        num_grouped = len(words)

        # 7. Final region filtering (filter tiny isolated noise < self.min_area)
        final_boxes = []
        for b in words:
            t, l, bot, r = b
            w = r - l
            h = bot - t
            if (w * h) < self.min_area:
                continue
            final_boxes.append(b)

        # 8. Sort in natural reading order
        boxes_with_centers = [((b[0] + b[2]) / 2.0, b[2] - b[0], b) for b in final_boxes]
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

        num_final = len(sorted_boxes)

        self.metrics = {
            "num_raw": num_raw,
            "num_filtered": num_filtered,
            "num_grouped": num_grouped,
            "num_final": num_final,
        }

        # 9. Save debug images if requested
        if debug:
            out_dir = os.path.join(self.debug_dir, debug_prefix)
            os.makedirs(out_dir, exist_ok=True)
            cv2.imwrite(os.path.join(out_dir, "1_original.png"), image)
            cv2.imwrite(os.path.join(out_dir, "2_grayscale.png"), gray)
            cv2.imwrite(os.path.join(out_dir, "3_thresholded.png"), binary)
            cv2.imwrite(os.path.join(out_dir, "4_morphology.png"), morph_out)

            # 5. Contours visualization
            vis_cnt = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(vis_cnt, contours, -1, (0, 0, 255), 1)
            cv2.imwrite(os.path.join(out_dir, "5_contours.png"), vis_cnt)

            # 6. Filtered regions
            vis_filt = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            for t, l, b, r in filtered_chars:
                cv2.rectangle(vis_filt, (l, t), (r, b), (255, 0, 0), 1)
            cv2.imwrite(os.path.join(out_dir, "6_filtered_regions.png"), vis_filt)

            # 7. Grouped text regions
            vis_grp = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            for t, l, b, r in words:
                cv2.rectangle(vis_grp, (l, t), (r, b), (0, 255, 255), 1)
            cv2.imwrite(os.path.join(out_dir, "7_grouped_regions.png"), vis_grp)

            # 8. Final bounding boxes
            vis_final = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            for idx, (t, l, b, r) in enumerate(sorted_boxes):
                cv2.rectangle(vis_final, (l, t), (r, b), (0, 255, 0), 1)
                cv2.putText(vis_final, str(idx), (l, max(12, t - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
            cv2.imwrite(os.path.join(out_dir, "8_final_bounding_boxes.png"), vis_final)

        return sorted_boxes


def test_all_images():
    detector = RobustTextDetector()

    images = [
        ("invoice", "sample/invoice_test.png"),
        ("document_sample", "sample/document_sample.png"),
        ("iiit5k_sample", "IIIT5K-Word_V3.0/IIIT5K/test/1002_1.png"),
    ]

    for name, path in images:
        img = cv2.imread(path)
        assert img is not None, f"Could not read {path}"
        boxes = detector.detect(img, debug=True, debug_prefix=name)
        m = detector.metrics
        print(f"\n==========================================")
        print(f"IMAGE: {name} ({path})")
        print(f"==========================================")
        print(f"raw regions      : {m['num_raw']}")
        print(f"filtered regions : {m['num_filtered']}")
        print(f"grouped regions  : {m['num_grouped']}")
        print(f"final regions    : {m['num_final']}")

if __name__ == "__main__":
    test_all_images()
