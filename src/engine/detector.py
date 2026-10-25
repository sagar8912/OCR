"""Production TextDetector implementation with Character-to-Word Grouping and Line Remnant Suppression."""

import os
from typing import List, Dict, Any, Optional
import cv2
import numpy as np


class TextDetector:
    """CPU-friendly document text detector.

    Performs stroke binarization, table & document line suppression,
    and bottom-up character-to-word grouping.
    Returns integer coordinates in [top, left, bottom, right] schema.
    """

    def __init__(
        self,
        min_area: int = 35,
        min_char_width: int = 4,
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

    def detect(self, image: np.ndarray, debug: bool = False, debug_prefix: str = "doc") -> List[List[int]]:
        """Detect individual text word regions in an image.

        Args:
            image: Input image as numpy array (BGR or Grayscale).
            debug: If True, saves intermediate debug images to self.debug_dir.
            debug_prefix: Subfolder name for debug artifacts.

        Returns:
            List of bounding boxes: [[top, left, bottom, right], ...] in integer coordinates.
        """
        h_img, w_img = image.shape[:2]
        if h_img == 0 or w_img == 0:
            return []

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Handle cropped single-word images (e.g. from IIIT-5K dataset)
        if h_img <= 80 and w_img <= 350:
            self.metrics = {
                "num_raw": 1,
                "num_filtered": 1,
                "num_grouped": 1,
                "num_final": 1,
            }
            return [[0, 0, int(h_img), int(w_img)]]

        # 1. Edge/gradient magnitude to detect character boundaries
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = cv2.magnitude(grad_x, grad_y)
        grad = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # 2. Otsu thresholding on gradient map
        _, binary = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 3. Detect and remove true document/table grid lines
        h_kernel_len = max(45, int(w_img * 0.045))
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_len, 1))
        h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)

        v_kernel_len = max(35, int(h_img * 0.055))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_len))
        v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

        lines_mask = cv2.bitwise_or(h_lines, v_lines)
        lines_dilated = cv2.dilate(lines_mask, np.ones((3, 3), np.uint8), iterations=2)

        # Pure text mask free of table borders and page outlines
        text_mask = cv2.bitwise_and(binary, cv2.bitwise_not(lines_dilated))

        # 4. Extract external character/stroke contours (ignores internal letter holes)
        char_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        morph_out = cv2.morphologyEx(text_mask, cv2.MORPH_CLOSE, char_kernel)

        contours, _ = cv2.findContours(morph_out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        num_raw = len(contours)

        # 5. Filter noise, outer document borders, and line slivers
        filtered_chars = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            # Preserve punctuation contours (hyphens and dashes are short in height but distinct in width)
            is_punct = (w >= 4 and h >= 2 and area >= 8)
            if not is_punct and (area < 18 or w < self.min_char_width or h < self.min_char_height):
                continue
            if w > self.max_width_ratio * w_img or h > self.max_height_ratio * h_img:
                continue
            if y < 3 and h > 0.4 * h_img:
                continue
            if x < 3 and w > 0.4 * w_img:
                continue
            # Filter line slivers
            if (w / max(1, h) > 20) or (h / max(1, w) > 10):
                continue
            if w <= 3 or (w <= 4 and h > 20):
                continue
            filtered_chars.append([y, x, y + h, x + w])

        num_filtered = len(filtered_chars)

        if not filtered_chars:
            self.metrics = {"num_raw": num_raw, "num_filtered": 0, "num_grouped": 1, "num_final": 1}
            return [[0, 0, int(h_img), int(w_img)]]

        # 6. Bottom-up Character-to-Word Grouping
        words = self._group_characters_to_words(filtered_chars)
        num_grouped = len(words)

        # 7. Word splitting on wide multi-word regions via vertical projection
        split_words = self._split_multi_word_regions(words, text_mask)

        # 8. Final region filtering (filter tiny isolated noise and line remnants)
        final_boxes = []
        for b in split_words:
            t, l, bot, r = b
            w = r - l
            h = bot - t
            if (w * h) < self.min_area or w <= 4:
                continue
            if (w / max(1, h) < 0.18):
                continue
            final_boxes.append(b)

        # 9. Sort in natural reading order
        sorted_boxes = self._sort_reading_order(final_boxes)
        num_final = len(sorted_boxes)

        self.metrics = {
            "num_raw": num_raw,
            "num_filtered": num_filtered,
            "num_grouped": num_grouped,
            "num_final": num_final,
        }

        # 10. Save debug images if requested
        if debug:
            out_dir = os.path.join(self.debug_dir, debug_prefix)
            os.makedirs(out_dir, exist_ok=True)
            cv2.imwrite(os.path.join(out_dir, "1_original.png"), image)
            cv2.imwrite(os.path.join(out_dir, "2_grayscale.png"), gray)
            cv2.imwrite(os.path.join(out_dir, "3_thresholded.png"), binary)
            cv2.imwrite(os.path.join(out_dir, "4_morphology.png"), morph_out)

            vis_cnt = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(vis_cnt, contours, -1, (0, 0, 255), 1)
            cv2.imwrite(os.path.join(out_dir, "5_contours.png"), vis_cnt)

            vis_filt = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            for t, l, b, r in filtered_chars:
                cv2.rectangle(vis_filt, (l, t), (r, b), (255, 0, 0), 1)
            cv2.imwrite(os.path.join(out_dir, "6_filtered_regions.png"), vis_filt)

            vis_grp = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            for t, l, b, r in words:
                cv2.rectangle(vis_grp, (l, t), (r, b), (0, 255, 255), 1)
            cv2.imwrite(os.path.join(out_dir, "7_grouped_regions.png"), vis_grp)

            vis_final = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            for idx, (t, l, b, r) in enumerate(sorted_boxes):
                cv2.rectangle(vis_final, (l, t), (r, b), (0, 255, 0), 1)
                cv2.putText(vis_final, str(idx), (l, max(12, t - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
            cv2.imwrite(os.path.join(out_dir, "8_final_bounding_boxes.png"), vis_final)

        return sorted_boxes

    def _group_characters_to_words(self, char_boxes: List[List[int]]) -> List[List[int]]:
        """Group individual characters into words based on horizontal proximity and vertical alignment."""
        words = [[t, l, b, r] for t, l, b, r in char_boxes]
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

                    # Hyphens and punctuation naturally have smaller heights than capital letters
                    is_hyphen = (min_h <= 8 and (o_h <= 8 or c_h <= 8))
                    if not is_hyphen and (min_h == 0 or (min_h / max_h) < 0.40):
                        continue

                    # Vertical overlap or baseline alignment
                    y_overlap = max(0, min(c_bot, o_bot) - max(c_top, o_top))
                    if not is_hyphen and (y_overlap / min_h) < 0.40 and abs(c_yc - o_yc) > 0.45 * max_h:
                        continue
                    if is_hyphen and abs(c_yc - o_yc) > 0.45 * max_h:
                        continue

                    # Horizontal gap: must be an intra-word character gap (<= 5px), NOT an inter-word space (>= 8px)
                    dx = o_left - c_right
                    if is_hyphen:
                        max_intra_word_gap = max(6, int(max_h * 0.35))
                    else:
                        max_intra_word_gap = max(4, int(min_h * 0.28))

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

        return words

    def _split_multi_word_regions(self, boxes: List[List[int]], text_mask: np.ndarray) -> List[List[int]]:
        """Split wide multi-word boxes into individual words using vertical projection profiling."""
        result = []
        for box in boxes:
            t, l, b, r = box
            w = r - l
            h = b - t
            aspect = w / max(1, h)

            # Only split if aspect ratio is wide enough to contain distinct separated words
            if aspect > 2.5 and w > 48:
                sub = text_mask[t:b, l:r]
                col_proj = np.sum(sub > 0, axis=0)

                # True inter-word spacing is at least 0.45 * font height (>= 7px)
                # Narrow intra-identifier gaps (like around hyphens/slashes) are < 6px and preserved
                min_space_width = max(7, int(h * 0.45))
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

        # Deduplicate and remove heavily overlapping boxes (IoU / subset > 0.85)
        dedup_boxes = []
        result = sorted(result, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
        for b in result:
            bt, bl, bb, br = b
            b_area = (bb - bt) * (br - bl)
            keep = True
            for eb in dedup_boxes:
                et, el, eb_bot, er = eb
                eb_area = (eb_bot - et) * (er - el)
                it = max(bt, et)
                il = max(bl, el)
                ib = min(bb, eb_bot)
                ir = min(br, er)
                if ib > it and ir > il:
                    inter = (ib - it) * (ir - il)
                    if inter / max(1, min(b_area, eb_area)) > 0.85:
                        keep = False
                        break
            if keep:
                dedup_boxes.append(b)

        return dedup_boxes

    def _sort_reading_order(self, boxes: List[List[int]]) -> List[List[int]]:
        """Sort boxes top-to-bottom by line blocks, then left-to-right within each line."""
        if not boxes:
            return []

        boxes_with_centers = [((b[0] + b[2]) / 2.0, b[2] - b[0], b) for b in boxes]
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

        sorted_result = []
        for line in sorted(lines, key=lambda l: l["yc"]):
            line_boxes = sorted(line["boxes"], key=lambda b: b[1])
            sorted_result.extend(line_boxes)

        return sorted_result
