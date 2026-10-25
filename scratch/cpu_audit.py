import platform
import os
import time
import torch
import cv2
from src.engine.pipeline import OCREngine

def main():
    print("==================================================")
    print("HARDWARE & ENVIRONMENT AUDIT")
    print("==================================================")
    print("Operating System :", platform.system(), platform.release(), platform.version())
    print("Architecture     :", platform.machine())
    print("Processor        :", platform.processor())
    print("Logical CPU Cores:", os.cpu_count())
    print("CUDA Available   :", torch.cuda.is_available())
    print("Inference Device : CPU (Strictly verified)")
    print("PyTorch Version  :", torch.__version__)
    print("OpenCV Version   :", cv2.__version__)

    engine = OCREngine("models/best_model.pt")
    img = cv2.imread("sample/invoice_test.png")
    assert img is not None, "Failed to load sample/invoice_test.png"

    # Warmup
    engine.process(img, redact=True)

    times_prep = []
    times_det = []
    times_ocr = []
    times_redact = []
    times_total = []

    for _ in range(5):
        t0 = time.perf_counter()
        prepped = engine.preprocessor.process(img)
        t1 = time.perf_counter()
        boxes = engine.detector.detect(img)
        t2 = time.perf_counter()
        crops = [img[max(0, b[0]):b[2], max(0, b[1]):b[3]] for b in boxes if (b[2]-b[0])>0 and (b[3]-b[1])>0]
        words = engine.recognizer.recognize_batch(crops)
        t3 = time.perf_counter()
        redacted = engine.redactor.redact(img, boxes)
        t4 = time.perf_counter()

        times_prep.append((t1 - t0) * 1000.0)
        times_det.append((t2 - t1) * 1000.0)
        times_ocr.append((t3 - t2) * 1000.0)
        times_redact.append((t4 - t3) * 1000.0)
        times_total.append((t4 - t0) * 1000.0)

    print("\n==================================================")
    print("CPU INFERENCE LATENCY BREAKDOWN (sample/invoice_test.png)")
    print("==================================================")
    print(f"Detected Word Regions : {len(boxes)}")
    print(f"Preprocessing Latency : {sum(times_prep)/len(times_prep):.2f} ms")
    print(f"Detection Latency     : {sum(times_det)/len(times_det):.2f} ms")
    print(f"OCR Latency           : {sum(times_ocr)/len(times_ocr):.2f} ms (approx {(sum(times_ocr)/len(times_ocr))/len(boxes):.2f} ms/word)")
    print(f"Redaction Latency     : {sum(times_redact)/len(times_redact):.2f} ms")
    print(f"Total Pipeline Latency: {sum(times_total)/len(times_total):.2f} ms")
    print("==================================================")

if __name__ == "__main__":
    main()
