import cv2
import numpy as np
from facepybird import FacePyBird

if __name__ == '__main__':
    capture: cv2.VideoCapture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    facepyb: FacePyBird = FacePyBird(True)

    while True:
        if cv2.waitKey(20) == 27:
            break

        frame: np.ndarray = capture.read()[1]
        frame = cv2.flip(frame, 1)

        facepyb.get_frame(frame)
        cv2.imshow('FacePy Bird OpenCV by @victor-hugo-dc', frame)
    
    capture.release()
    cv2.destroyAllWindows()