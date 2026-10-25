import cv2
from src.engine.detector import TextDetector

def main():
    img = cv2.imread("sample/invoice_test.png")
    detector = TextDetector()
    boxes = detector.detect(img)

    print("Boxes in Invoice No / Date / Customer region (y in [80, 160], x > 600):")
    for b in boxes:
        top, left, bot, right = b
        if 70 <= top <= 165 and left >= 600:
            print(f"  Box: {b} -> h={bot-top}, w={right-left}")

    print("\nBoxes in Address region (y in [80, 120], x in [40, 300]):")
    for b in boxes:
        top, left, bot, right = b
        if 70 <= top <= 115 and left < 350:
            print(f"  Box: {b} -> h={bot-top}, w={right-left}")

    print("\nBoxes in Currency / Table Amounts (y in [340, 580], x > 650):")
    for b in boxes:
        top, left, bot, right = b
        if 340 <= top <= 580 and left >= 650:
            print(f"  Box: {b} -> h={bot-top}, w={right-left}")

if __name__ == "__main__":
    main()
