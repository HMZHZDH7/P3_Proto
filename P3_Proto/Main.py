import cv2
import numpy as np
from P3_Proto.image_Processor import*

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    IP = imageProcessor(frame)
    mask = IP.image_to_binary(frame)
    mask_Closed = IP.reduce_noise(mask)

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

    Blobs = detector.detect(mask_Closed)
    print(Blobs)

    pts = cv2.KeyPoint_convert(Blobs)

    if len(pts) == 2:
        if pts[0, 0] > pts[1, 0]:
            left_hand = pts[0, 0]
            right_hand = pts[1, 0]
        else:
            left_hand = pts[1, 0]
            right_hand = pts[0, 0]

    im_with_keypoints = cv2.drawKeypoints(mask_Closed, Blobs, np.array([]), (0, 0, 255),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow('frame', frame)
    cv2.imshow('frame4', im_with_keypoints)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
