from eigenfaces import FrameProcessor
import numpy as np
import cv2

if __name__ == "__main__":
    recognizer = FrameProcessor()

    video_capture = cv2.VideoCapture(0)
    while True:
        ret, frame = video_capture.read()
        in_img = np.array(frame)
        out_img = recognizer.recognize(in_img)
        cv2.imshow("Video", out_img)

        # When everything is done, release the capture on pressing "q"
        if cv2.waitKey(1) & 0xFF == ord("q"):
            video_capture.release()
            cv2.destroyAllWindows()