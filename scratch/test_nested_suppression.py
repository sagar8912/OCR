"""Test nested contour suppression."""

def suppress_nested_boxes(boxes):
    # Sort boxes by area descending
    boxes = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    kept = []
    
    for b in boxes:
        t, l, bot, r = b
        area = (bot - t) * (r - l)
        is_nested = False
        
        for k in kept:
            kt, kl, kbot, kr = k
            # compute intersection
            int_t = max(t, kt)
            int_l = max(l, kl)
            int_bot = min(bot, kbot)
            int_r = min(r, kr)
            
            if int_bot > int_t and int_r > int_l:
                int_area = (int_bot - int_t) * (int_r - int_l)
                # If b is mostly contained within k
                if (int_area / float(area)) > 0.65:
                    is_nested = True
                    break
        
        if not is_nested:
            kept.append(b)
            
    return kept

import cv2
from scratch.test_char_grouping import run_char_to_word_grouping
