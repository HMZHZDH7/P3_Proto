import cv2
import numpy as np


class imageProcessor:
    power = 0.0
    distance = 0.0
    pos_left_hand = 0.0
    pos_right_hand = 0.0
    binary_Image = None

    def __init__(self, frame):
        self.binary_Image = frame

    def image_to_binary(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (50, 25, 25), (100, 200, 200))
        return mask

    def reduce_noise(self, image):
        kernel = np.ones((5, 5), np.uint8)

        mask_opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        mask_closed = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel)
        return mask_closed
