"""Run detector metrics test."""
import cv2
from scratch.refined_detector_class import TextDetector

detector = TextDetector()
for p in ['sample/invoice_test.png', 'sample/document_sample.png']:
    img = cv2.imread(p)
    boxes = detector.detect(img, debug=True)
    m = detector.last_metrics
    print(f"=== {p} ===")
    print(f"  number of raw contours/components : {m['num_raw']}")
    print(f"  number after filtering            : {m['num_filtered']}")
    print(f"  number after grouping             : {m['num_merged']}")
    print(f"  final number of text regions      : {m['num_final']}")
