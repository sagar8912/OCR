import os
import json
import time
from pathlib import Path
import cv2
import numpy as np
from src.engine.pipeline import OCREngine

def analyze_redaction_quality(orig_bgr: np.ndarray, redacted_bgr: np.ndarray, boxes: list) -> dict:
    """Analyze how cleanly text strokes were removed in redacted bounding boxes."""
    if not boxes:
        return {"avg_stroke_reduction_pct": 100.0, "cleanly_redacted_boxes": 0, "total_boxes": 0}

    orig_gray = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2GRAY)
    red_gray = cv2.cvtColor(redacted_bgr, cv2.COLOR_BGR2GRAY)

    # Edge density before vs after
    orig_edges = cv2.Canny(orig_gray, 50, 150)
    red_edges = cv2.Canny(red_gray, 50, 150)

    stroke_reductions = []
    for top, left, bottom, right in boxes:
        top = max(0, int(top))
        left = max(0, int(left))
        bottom = min(orig_bgr.shape[0], int(bottom))
        right = min(orig_bgr.shape[1], int(right))

        if bottom <= top or right <= left:
            continue

        orig_patch_edges = orig_edges[top:bottom, left:right]
        red_patch_edges = red_edges[top:bottom, left:right]

        orig_count = np.count_nonzero(orig_patch_edges)
        red_count = np.count_nonzero(red_patch_edges)

        reduction = 100.0 if orig_count == 0 else max(0.0, (1.0 - (red_count / orig_count)) * 100.0)
        stroke_reductions.append(reduction)

    avg_reduction = np.mean(stroke_reductions) if stroke_reductions else 100.0
    return {
        "avg_stroke_reduction_pct": round(float(avg_reduction), 2),
        "total_boxes_evaluated": len(stroke_reductions),
        "perfectly_cleared_pct": round(float(np.mean([1.0 if r > 85.0 else 0.0 for r in stroke_reductions]) * 100.0), 2)
    }

def main():
    engine = OCREngine("models/best_model.pt")

    test_images = [
        {"name": "Invoice Image 1", "path": "sample/invoice_test.png"},
        {"name": "Invoice Image 2", "path": "sample/invoice_sample_2.png"},
        {"name": "IIIT-5K Sample", "path": "IIIT5K-Word_V3.0/IIIT5K/test/1002_2.png"},
        {"name": "Document Sample", "path": "sample/document_sample.png"},
    ]

    results = []

    print("==================================================")
    print("RUNNING MULTI-IMAGE DETECTION, OCR & REDACTION AUDIT")
    print("==================================================")

    for item in test_images:
        path = Path(item["path"])
        if not path.exists():
            print(f"File not found: {path}")
            continue

        orig_bgr = cv2.imread(str(path))
        h, w = orig_bgr.shape[:2]

        t0 = time.perf_counter()
        proc_res = engine.process(path, redact=True)
        latency = (time.perf_counter() - t0) * 1000.0

        outputs = engine.save_outputs(proc_res, path)
        words_data = proc_res["results"]
        boxes = [w_item["bounding_box"] for w_item in words_data]

        # Check border suppression: ensure NO bounding box covers > 80% of image area
        max_area_ratio = max([(b[2]-b[0])*(b[3]-b[1])/(h*w) for b in boxes]) if boxes else 0.0
        border_suppressed = max_area_ratio < 0.80

        # Check redaction quality
        redaction_eval = analyze_redaction_quality(orig_bgr, proc_res["redacted_image"], boxes)

        # Collect sample transcriptions
        sample_words = [w_item["word"] for w_item in words_data[:8]]

        img_report = {
            "name": item["name"],
            "path": str(path),
            "dimensions": f"{w}x{h}",
            "detected_boxes": len(boxes),
            "max_box_area_ratio": round(max_area_ratio, 4),
            "border_suppression_pass": border_suppressed,
            "redaction_metrics": redaction_eval,
            "latency_ms": round(latency, 2),
            "saved_json": outputs["json_path"],
            "saved_redacted": outputs["redacted_image_path"],
            "sample_transcriptions": sample_words,
        }
        results.append(img_report)

        print(f"\n--- {item['name']} ({path.name}) ---")
        print(f"Dimensions           : {w}x{h}")
        print(f"Detected Word Regions: {len(boxes)}")
        print(f"Border Suppression   : {'PASS (No giant box)' if border_suppressed else 'FAIL'}")
        print(f"Redaction Edge Cleared: {redaction_eval['avg_stroke_reduction_pct']}% reduction in text edges")
        print(f"Redacted Image Saved : {outputs['redacted_image_path']}")
        print(f"JSON Results Saved   : {outputs['json_path']}")
        print(f"Latency              : {latency:.2f} ms")
        print(f"Sample Transcriptions: {sample_words}")

    with open("scratch/multi_image_audit_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n==================================================")
    print("AUDIT EXECUTION COMPLETE. Results saved to scratch/multi_image_audit_results.json")
    print("==================================================")

if __name__ == "__main__":
    main()
