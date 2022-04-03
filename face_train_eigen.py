import numpy as np
import cv2
import sys
import os

FREQ_DIV = 5   #frequency divider for capturing training images
RESIZE_FACTOR = 4
NUM_TRAINING = 5

class TrainEigenFaces:
    def __init__(self):
        cascPath = "models/haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascPath)
        self.face_dir = "data"
        self.face_name = sys.argv[1]
        self.path = os.path.join(self.face_dir, self.face_name)
        if not os.path.isdir(self.path):
            os.makedirs(self.path)
        self.model = cv2.face.EigenFaceRecognizer_create()
        self.count_captures = 0
        self.count_timer = 0

    def capture_training_images(self):
        video_capture = cv2.VideoCapture(0)
        while True:
            ret, frame = video_capture.read()
            frame_flipped = cv2.flip(np.array(frame), 1)

            k = cv2.waitKey(100) & 0xff # press "ESC" for exiting video
            if k == 27:
                break
            elif self.count_captures == NUM_TRAINING:
                break
            elif k == 13:
                face, coords = self.process_image(frame_flipped)
                if face is not None:
                    (x, y, w, h) = coords
                    img_no = sorted([int(fn[:fn.find(".")]) for fn in os.listdir(self.path) if fn[0]!="." ]+[0])[-1] + 1

                    cv2.imwrite("%s/%s.png" % (self.path, img_no), face)
                    self.count_captures += 1
                    print(f"Captured image: {self.count_captures}")

                    cv2.rectangle(frame_flipped, (x, y), (x+w, y+h), (0, 255, 0), 3)
                    cv2.putText(frame_flipped, self.face_name, (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1,(0, 255, 0))

            cv2.imshow("image", frame_flipped)

        video_capture.release()
        cv2.destroyAllWindows()
        return

    def process_image(self, frame):
        resized_width, resized_height = (100, 100)    
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        gray_resized = cv2.resize(gray, (int(gray.shape[1]/RESIZE_FACTOR), int(gray.shape[0]/RESIZE_FACTOR)))        
        faces = self.face_cascade.detectMultiScale(
            gray_resized,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
            )
        if len(faces) > 0:
            areas = []
            for (x, y, w, h) in faces: 
                areas.append(w*h)
            max_area, idx = max([(val,idx) for idx,val in enumerate(areas)])
            face_sel = faces[idx]

            x = face_sel[0] * RESIZE_FACTOR
            y = face_sel[1] * RESIZE_FACTOR
            w = face_sel[2] * RESIZE_FACTOR
            h = face_sel[3] * RESIZE_FACTOR

            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (resized_width, resized_height))

            return face_resized, (x, y, w, h)
        return None, None

    def eigen_train_data(self):
        imgs = []
        tags = []
        index = 0

        for (subdirs, dirs, files) in os.walk(self.face_dir):
            for subdir in dirs:
                img_path = os.path.join(self.face_dir, subdir)
                for fn in os.listdir(img_path):
                    path = img_path + "/" + fn
                    tag = index
                    imgs.append(cv2.imread(path, 0))
                    tags.append(int(tag))
                index += 1
        (imgs, tags) = [np.array(item) for item in [imgs, tags]]

        self.model.train(imgs, tags)
        self.model.save("models/eigen_trained_data.xml")
        print("Training completed successfully")
        return


if __name__ == "__main__":
    trainer = TrainEigenFaces()
    trainer.capture_training_images()
    trainer.eigen_train_data()
