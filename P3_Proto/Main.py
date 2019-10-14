import cv2
import numpy as np
from P3_Proto.image_Processor import*

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    IP = imageProcessor(frame)
    mask = IP.create_mask(frame)
    mask_Closed = IP.reduce_noise(mask)
    bloby_image = IP.detect_blobs(mask_Closed)

    pts = cv2.KeyPoint_convert(bloby_image)

    if len(pts) == 2:
        if pts[0, 0] > pts[1, 0]:
            left_hand = pts[0, 0]
            right_hand = pts[1, 0]
        else:
            left_hand = pts[1, 0]
            right_hand = pts[0, 0]

    im_with_keypoints = cv2.drawKeypoints(mask_Closed, bloby_image, np.array([]), (0, 0, 255),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow('frame', frame)
    cv2.imshow('frame4', im_with_keypoints)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
