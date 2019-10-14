import cv2
import numpy as np


class imageProcessor:
    power = 0.0
    distance = 0.0
    pos_left_hand = 0.0
    pos_right_hand = 0.0
    frame = None
    mask = None

    def __init__(self, frame):
        self.power = 0.0
        self.distance = 0.0
        self.pos_left_hand = 0.0
        self.pos_right_hand = 0.0
        self.frame = frame

    def create_mask(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (50, 25, 25), (100, 200, 200))
        return mask

    def reduce_noise(self, image):
        kernel = np.ones((5, 5), np.uint8)
        mask_opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        mask_closed = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel)
        self.mask = mask_closed
        return mask_closed

    def detect_blobs(self, image):
        # Setup SimpleBlobDetector parameters.
        params = cv2.SimpleBlobDetector_Params()

        params.filterByArea = True
        params.minArea = 200
        params.filterByColor = True
        params.blobColor = 255
        params.filterByCircularity = False
        params.filterByConvexity = False
        params.filterByInertia = False

        # Create a detector with the parameters
        ver = cv2.__version__.split('.')
        if int(ver[0]) < 3:
            detector = cv2.SimpleBlobDetector(params)
        else:
            detector = cv2.SimpleBlobDetector_create(params)

        blobs = detector.detect(image)
        return blobs

    def locate_hands(self, image):
        pts = cv2.KeyPoint_convert(image)
        if len(pts) == 2:
            if pts[0, 0] > pts[1, 0]:
                self.pos_left_hand = pts[0, 0]
                self.pos_right_hand = pts[1, 0]
            else:
                self.pos_left_hand = pts[1, 0]
                self.pos_right_hand = pts[0, 0]
