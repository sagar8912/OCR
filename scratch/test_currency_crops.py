import cv2
import numpy as np
from src.engine.decoder import analyze_crop_topology, classify_field_type

img = cv2.imread("sample/invoice_test.png")
boxes = [
    [355, 699, 371, 754], # 45,000
    [355, 919, 371, 974], # 90,000
    [389, 711, 406, 754], # 1,200
    [389, 929, 406, 974], # 6,000
    [423, 711, 440, 754], # 1,500
    [423, 930, 440, 974], # 7,500
    [457, 701, 474, 754], # 12,000
    [457, 919, 474, 974], # 24,000
    [491, 912, 508, 974], # 127,500
    [525, 920, 542, 974], # 22,950
    [561, 907, 578, 974], # 150,450
]

for i, b in enumerate(boxes):
    crop = img[b[0]:b[2], b[1]:b[3]]
    topo = analyze_crop_topology(crop)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    # Check connected components in crop
    _, bin_img = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_img)
    comp_list = []
    for c_i in range(1, num_labels):
        x, y, cw, ch, area = stats[c_i]
        comp_list.append((x, y, cw, ch, area))
    print(f"[{i:02d}] Box {b} (w={w}, h={h}): has_comma={topo['has_comma']}, commas={topo['commas']}, num_comps={num_labels-1}")
    for c in comp_list:
        print(f"     Comp: x={c[0]}, y={c[1]}, w={c[2]}, h={c[3]}, area={c[4]}")
