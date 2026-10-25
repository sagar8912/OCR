"""Text redaction module using OpenCV Telea and Navier-Stokes inpainting."""

from typing import List, Union
import cv2
import numpy as np


class TextRedactor:
    """Inpainting-based text redaction engine.

    Removes detected text bounding boxes and reconstructs surrounding background seamlessly
    using Navier-Stokes or Alexandru Telea fast marching inpainting methods.
    """

    def __init__(
        self,
        method: str = "telea",
        inpaint_radius: int = 5,
        padding: int = 2,
    ):
        """Initialize redactor.

        Args:
            method: 'telea' (cv2.INPAINT_TELEA) or 'ns' (cv2.INPAINT_NS).
            inpaint_radius: Radius of circular neighborhood considered for inpainting.
            padding: Pixel padding added around bounding boxes to mask character strokes completely.
        """
        self.method_name = method.lower()
        if self.method_name == "ns":
            self.inpaint_flag = cv2.INPAINT_NS
        else:
            self.inpaint_flag = cv2.INPAINT_TELEA

        self.inpaint_radius = inpaint_radius
        self.padding = padding

    def create_mask(
        self,
        image_shape: tuple,
        bounding_boxes: List[List[int]],
    ) -> np.ndarray:
        """Create a binary redaction mask from bounding boxes.

        Args:
            image_shape: (H, W) or (H, W, C) of the target image.
            bounding_boxes: List of [top, left, bottom, right] coordinates.

        Returns:
            Single-channel uint8 mask (0 = preserve, 255 = inpaint/redact).
        """
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        for box in bounding_boxes:
            if len(box) != 4:
                continue
            top, left, bottom, right = box

            # Apply expansion padding to ensure character outer edges are fully covered
            top_pad = max(0, int(top) - self.padding)
            left_pad = max(0, int(left) - self.padding)
            bottom_pad = min(h, int(bottom) + self.padding)
            right_pad = min(w, int(right) + self.padding)

            if bottom_pad > top_pad and right_pad > left_pad:
                mask[top_pad:bottom_pad, left_pad:right_pad] = 255

        return mask

    def redact(
        self,
        image: np.ndarray,
        bounding_boxes: List[List[int]],
    ) -> np.ndarray:
        """Redact text regions from the image using inpainting.

        Args:
            image: Source image numpy array (BGR or Grayscale).
            bounding_boxes: List of [top, left, bottom, right] integer bounding boxes.

        Returns:
            Redacted image as a BGR numpy array with text inpaint-removed.
        """
        if not bounding_boxes:
            return image.copy()

        # OpenCV inpainting works best on 3-channel BGR images
        if len(image.shape) == 2:
            bgr_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            is_originally_gray = True
        else:
            bgr_image = image.copy()
            is_originally_gray = False

        # Generate inpaint mask
        mask = self.create_mask(bgr_image.shape, bounding_boxes)

        # Apply OpenCV Inpainting
        redacted = cv2.inpaint(
            bgr_image,
            mask,
            inpaintRadius=self.inpaint_radius,
            flags=self.inpaint_flag,
        )

        return redacted
