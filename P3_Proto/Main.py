import cv2
import numpy as np
from P3_Proto.image_Processor import*

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    IP = imageProcessor(frame)
    IP.binary_Image = IP.create_mask(frame)
    mask_Closed = IP.reduce_noise(IP.binary_Image)
    bloby_image = IP.detect_blobs(mask_Closed)
    IP.locate_hands(bloby_image)

    im_with_keypoints = cv2.drawKeypoints(mask_Closed, bloby_image, np.array([]), (0, 0, 255),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow('frame', frame)
    cv2.imshow('frame4', im_with_keypoints)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
