"""Test vertical projection profile to split multi-word lines into individual words."""
import cv2
import numpy as np

img = cv2.imread("sample/invoice_test.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Sobel gradient
grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
grad = cv2.magnitude(grad_x, grad_y)
grad = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
_, binary = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Remove lines
h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1)))
v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25)))
lines = cv2.dilate(cv2.bitwise_or(h_lines, v_lines), np.ones((3, 3), np.uint8))
text_mask = cv2.bitwise_and(binary, cv2.bitwise_not(lines))

# Test on "Total Amount": y: 557..576, x: 628..777
def split_line_into_words(line_mask, ymin, xmin, ymax, xmax):
    sub = line_mask[ymin:ymax, xmin:xmax]
    h, w = sub.shape
    if h == 0 or w == 0:
        return []
    
    # 1D column profile: number of text pixels in each column
    col_proj = np.sum(sub > 0, axis=0)
    
    # Smooth slightly with 3-wide window
    # Find zero or near-zero valleys
    words = []
    in_word = False
    start_x = 0
    min_space_width = max(4, int(h * 0.35))
    space_count = 0
    
    for x in range(w):
        val = col_proj[x]
        if val > 0:
            if not in_word:
                start_x = max(0, x - 1)
                in_word = True
            space_count = 0
        else:
            if in_word:
                space_count += 1
                if space_count >= min_space_width:
                    end_x = x - space_count + 1
                    if (end_x - start_x) >= 6:
                        words.append([ymin, xmin + start_x, ymax, xmin + end_x])
                    in_word = False
                    space_count = 0
    
    if in_word:
        end_x = w
        if (end_x - start_x) >= 6:
            words.append([ymin, xmin + start_x, ymax, xmin + end_x])
            
    return words

print("Test splitting 'Total Amount' [557, 628, 576, 777]:")
splits = split_line_into_words(text_mask, 557, 628, 576, 777)
for s in splits:
    print(f"  Word box: {s}, width={s[3]-s[1]}")

print("\nTest splitting 'Laptop Dell Inspiron' [353, 174, 372, 333]:")
splits = split_line_into_words(text_mask, 353, 174, 372, 333)
for s in splits:
    print(f"  Word box: {s}, width={s[3]-s[1]}")
