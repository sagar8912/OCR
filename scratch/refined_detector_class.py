"""Refined detector implementation with debug mode and stage reports."""

import os
from typing import List, Dict, Any, Tuple
import cv2
import numpy as np


class TextDetector:
    """CPU-efficient classical text region detector.

    Separates text strokes from document/table lines, prevents over-fragmentation,
    groups characters into complete words, and returns bounding boxes in [top, left, bottom, right] format.
    """

    def __init__(
        self,
        min_area: int = 40,
        min_width: int = 8,
        min_height: int = 7,
        max_width_ratio: float = 0.85,
        max_height_ratio: float = 0.35,
        debug_dir: str = "sample/debug_detector",
    ):
        self.min_area = min_area
        self.min_width = min_width
        self.min_height = min_height
        self.max_width_ratio = max_width_ratio
        self.max_height_ratio = max_height_ratio
        self.debug_dir = debug_dir
        self.last_metrics: Dict[str, int] = {}

    def detect(self, image: np.ndarray, debug: bool = False) -> List[List[int]]:
        """Detect individual text word regions in an image.

        Args:
            image: Input image as numpy array (BGR or Grayscale).
            debug: If True, saves intermediate debug images to self.debug_dir.

        Returns:
            List of bounding boxes: [[top, left, bottom, right], ...] in integer coordinates.
        """
        h_img, w_img = image.shape[:2]
        if h_img == 0 or w_img == 0:
            return []

        # Convert to single channel grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Handle cropped single-word images (e.g. from IIIT-5K dataset)
        if h_img <= 80 and w_img <= 350:
            return [[0, 0, int(h_img), int(w_img)]]

        # 1. Edge/gradient magnitude to detect character boundaries
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = cv2.magnitude(grad_x, grad_y)
        grad = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # 2. Otsu thresholding on gradient map
        _, binary = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 3. Detect and remove true document/table grid lines
        # Structural elements must be substantially longer than single characters (I, l, 1, T)
        # to prevent erasing vertical strokes.
        h_kernel_len = max(40, int(w_img * 0.045))
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_len, 1))
        h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)

        v_kernel_len = max(35, int(h_img * 0.055))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_len))
        v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

        lines_mask = cv2.bitwise_or(h_lines, v_lines)
        lines_dilated = cv2.dilate(lines_mask, np.ones((3, 3), np.uint8), iterations=2)

        # Pure text mask free of table borders and page outlines
        text_mask = cv2.bitwise_and(binary, cv2.bitwise_not(lines_dilated))

        # 4. Word-level character grouping
        # Bridges intra-word character strokes while respecting inter-word boundaries
        kw = max(8, min(14, int(w_img * 0.012)))
        kh = max(2, int(h_img * 0.005))
        word_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
        connected = cv2.morphologyEx(text_mask, cv2.MORPH_CLOSE, word_kernel)

        # 5. Retrieve contours
        contours, _ = cv2.findContours(connected, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        num_raw = len(contours)

        # 6. Noise and boundary filtering
        filtered_boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h

            # Filter noise and page-level borders
            if area < self.min_area or w < self.min_width or h < self.min_height:
                continue
            if w > self.max_width_ratio * w_img or h > self.max_height_ratio * h_img:
                continue
            if y < 3 and h > 0.4 * h_img:
                continue
            if x < 3 and w > 0.4 * w_img:
                continue
            # Filter thin border slivers
            if (w / max(1, h) > 25) or (h / max(1, w) > 15):
                continue

            top = max(0, int(y))
            left = max(0, int(x))
            bottom = min(h_img, int(y + h))
            right = min(w_img, int(x + w))
            filtered_boxes.append([top, left, bottom, right])

        num_filtered = len(filtered_boxes)

        if not filtered_boxes:
            return [[0, 0, int(h_img), int(w_img)]]

        # 7. Merge adjacent fragments on same baseline (adaptive font height tolerance)
        merged_boxes = self._merge_boxes(filtered_boxes)
        num_merged = len(merged_boxes)

        # 8. Adaptive word splitting for wide multi-word regions via 1D vertical projection
        split_boxes = self._split_multi_word_boxes(merged_boxes, text_mask)
        num_final = len(split_boxes)

        # 9. Sort in natural reading order
        sorted_boxes = self._sort_reading_order(split_boxes)

        self.last_metrics = {
            "num_raw": num_raw,
            "num_filtered": num_filtered,
            "num_merged": num_merged,
            "num_final": num_final,
        }

        # Save debug artifacts if requested
        if debug:
            os.makedirs(self.debug_dir, exist_ok=True)
            cv2.imwrite(os.path.join(self.debug_dir, "1_grayscale.png"), gray)
            cv2.imwrite(os.path.join(self.debug_dir, "2_gradient.png"), grad)
            cv2.imwrite(os.path.join(self.debug_dir, "3_binary.png"), binary)
            cv2.imwrite(os.path.join(self.debug_dir, "4_lines_mask.png"), lines_mask)
            cv2.imwrite(os.path.join(self.debug_dir, "5_text_mask.png"), text_mask)
            cv2.imwrite(os.path.join(self.debug_dir, "6_connected.png"), connected)

            vis = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            for idx, (t, l, b, r) in enumerate(sorted_boxes):
                cv2.rectangle(vis, (l, t), (r, b), (0, 255, 0), 1)
                cv2.putText(vis, str(idx), (l, max(12, t - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
            cv2.imwrite(os.path.join(self.debug_dir, "7_final_boxes.png"), vis)

        return sorted_boxes

    def _merge_boxes(self, boxes: List[List[int]]) -> List[List[int]]:
        """Merge bounding boxes that lie on the same line and are adjacent fragments."""
        if len(boxes) <= 1:
            return boxes

        sorted_by_left = sorted(boxes, key=lambda b: (b[1], b[0]))
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
                    # Dynamic gap allowed: scales with font height (e.g., 5px for small font, 16px for header)
                    gap_allowed = max(5, int(min_h * 0.45))
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

        return merged

    def _split_multi_word_boxes(self, boxes: List[List[int]], text_mask: np.ndarray) -> List[List[int]]:
        """Split wide multi-word boxes into individual words using vertical projection profiling."""
        result = []
        for box in boxes:
            t, l, b, r = box
            w = r - l
            h = b - t
            aspect = w / max(1, h)

            # Only split boxes that are significantly wide (multi-word lines)
            if aspect > 2.8 and w > 55:
                sub = text_mask[t:b, l:r]
                col_proj = np.sum(sub > 0, axis=0)

                # True inter-word spacing is typically >= 0.50 * font height
                min_space_width = max(5, int(h * 0.50))
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
                    result.extend(words_in_box)
                else:
                    result.append(box)
            else:
                result.append(box)

        return result

    def _sort_reading_order(self, boxes: List[List[int]]) -> List[List[int]]:
        """Sort boxes top-to-bottom by line blocks, then left-to-right within each line."""
        if not boxes:
            return []

        boxes_with_centers = []
        for b in boxes:
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

        sorted_result = []
        for line in sorted(lines, key=lambda l: l["yc"]):
            line_boxes = sorted(line["boxes"], key=lambda b: b[1])
            sorted_result.extend(line_boxes)

        return sorted_result
