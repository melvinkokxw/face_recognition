import os
import sys

import cv2
import numpy as np
from PIL import Image

class DataSet:
    def __init__(self, dataset_path):
        self.images = []
        self.image_labels = []
        self.class_images_list = []

        for _, class_names, _ in os.walk(dataset_path):
            for class_name in class_names:
                class_path = os.path.join(dataset_path, class_name)
                class_samples_list = []

                for filename in os.listdir(class_path):
                    if filename != ".DS_Store":
                        try:
                            im = Image.open(os.path.join(class_path, filename))
                            self.images.append(np.asarray(im, dtype = np.uint8))
                            # adds each sample within a class to this List
                            class_samples_list.append(np.asarray(im, dtype = np.uint8))
                        except IOError as e:
                            errno, strerror = e.args
                            print(f"I/O error({errno}): {strerror}")
                        except:
                            print(f"Unexpected error: {sys.exc_info()[0]}")
                            raise

                # flattens each sample within a class and adds the array/vector to a class matrix
                class_samples_matrix = np.array([img.flatten("C")
                    for img in class_samples_list])

                # adds each class matrix to this MASTER List
                self.class_images_list.append(class_samples_matrix)

                self.image_labels.append(class_name)

class EigenFaces:
    def __init__(self):
        self.images = None
        self.mean_image = None
        self.image_labels = []
        self.class_images_list = []
        self.v = None # Eigenvectors for face recognition
        self.w = [] # Eigenvector coefficients for face recognition

        self.images_emotions = None
        self.mean_image_emotions = None
        self.image_labels_emotions = []
        self.class_images_list_emotions = []
        self.v_emotions = None # Eigenvectors for emotion recognition
        self.w_emotions = [] # Eigenvector coefficients for emotion recognition

    def load_dataset(self, dataset: DataSet, dataset_emotions: DataSet):
        self.images = np.array([img.flatten() for img in dataset.images])
        self.image_labels = dataset.image_labels
        self.class_images_list = dataset.class_images_list

        self.images_emotions = np.array([img.flatten() for img in dataset_emotions.images])
        self.image_labels_emotions = dataset_emotions.image_labels
        self.class_images_list_emotions = dataset_emotions.class_images_list

    def train(self):
        self.v, self.mean_image = self.pca(self.images)

        # Save eigenfaces as images
        dim = int(np.sqrt(self.mean_image.shape[0]))
        for i in range(self.v.shape[0]):
            Image.fromarray(self.v[i].T.reshape((dim, dim)), mode="L").save(f"eigenfaces/{i}.jpg")

        # Projecting each class sample (as class matrix) and then using the class average as the class weights for comparison with the Target image
        for class_sample in self.class_images_list:
            class_weights_vertex = self.project_image(class_sample)
            self.w.append(class_weights_vertex.mean(0))

        self.v_emotions, self.mean_image_emotions = self.pca(self.images_emotions)

        # Projecting each class sample (as class matrix) and then using the class average as the class weights for comparison with the Target image
        for class_sample in self.class_images_list_emotions:
            class_weights_vertex = self.project_image_emotions(class_sample)
            self.w_emotions.append(class_weights_vertex.mean(0))

    def project_image(self, X):
        X = X - self.mean_image
        return np.dot(X, self.v.T)

    def project_image_emotions(self, X):
        X = X - self.mean_image_emotions
        return np.dot(X, self.v_emotions.T)

    def predict_face(self, X):
        min_class = -1
        min_distance = np.finfo('float').max

        min_class_emotions = -1
        min_distance_emotions = np.finfo('float').max

        projected_target = self.project_image(X)
        projected_target_emotions = self.project_image_emotions(X)

        # delete last array item, it's nan
        projected_target = np.delete(projected_target, -1)
        projected_target_emotions = np.delete(projected_target_emotions, -1)

        for i in range(len(self.w)):
            distance = np.linalg.norm(projected_target - np.delete(self.w[i], -1))
            if distance < min_distance:
                min_distance = distance
                min_class = self.image_labels[i]

            for i in range(len(self.w_emotions)):
                distance_emotions = np.linalg.norm(projected_target_emotions - np.delete(self.w_emotions[i], -1))
                if distance_emotions < min_distance_emotions:
                    min_distance_emotions = distance_emotions
                    min_class_emotions = self.image_labels_emotions[i]

        print(min_class, min_distance, min_class_emotions, min_distance_emotions)        

        return min_class, min_distance, min_class_emotions, min_distance_emotions

    @staticmethod
    def pca(X):
        # get dimensions
        num_data, dim = X.shape

        # center data
        mean_X = X.mean(axis=0)
        X = X - mean_X

        if dim > num_data:
            # PCA - compact trick used
            C = np.dot(X,X.T) # covariance matrix
            e, v = np.linalg.eigh(C) # eigenvalues and eigenvectors
            w = np.dot(X.T, v).T # this is the compact trick
            e = e[e >= 0] # Drop negative eigenvalues
            S = np.sqrt(e[::-1]) # reverse since eigenvalues are in increasing order
            V = w[e.shape[0]-1::-1] # reverse since last eigenvectors are the ones we want

            # Normalise eigenvectors
            for i in range(V.shape[1]):
                V[:,i] /= S

            # Normal method but too slow
            # C = np.cov(X.T)
            # e, V = np.linalg.eigh(C)
            # V = V[:num_data]
        else:
            # PCA - SVD used
            U, S, V = np.linalg.svd(X)
            V = V[:num_data] # only makes sense to return the first num_data

        return V, mean_X

class FaceClassifier:
    def __init__(self):
        self.dataset = DataSet("data")
        self.dataset_emotion = DataSet("data_emotion")
        self.classifier = EigenFaces()
        self.classifier.load_dataset(self.dataset, self.dataset_emotion)
        self.classifier.train()

        self.RESIZE_FACTOR = 4
        self.MAX_DISTANCE = 3500

    def recognize(self, frame):
        frame = cv2.flip(frame, 1)
        resized_width, resized_height = (100, 100)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)        
        gray_resized = cv2.resize(gray, (int(gray.shape[1]/self.RESIZE_FACTOR), int(gray.shape[0]/self.RESIZE_FACTOR)))        
        faces = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml").detectMultiScale(
            gray_resized,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        for i in range(len(faces)):
            face_i = faces[i]
            x = face_i[0] * self.RESIZE_FACTOR
            y = face_i[1] * self.RESIZE_FACTOR
            w = face_i[2] * self.RESIZE_FACTOR
            h = face_i[3] * self.RESIZE_FACTOR
            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (resized_width, resized_height))
            face_flattened = np.array(Image.fromarray(face_resized)).flatten()
            min_class, min_distance, min_class_emo, min_distance_emo = self.classifier.predict_face(face_flattened)

            if min_distance > self.MAX_DISTANCE:
                min_class = "Unknown Person"
            if min_distance_emo > self.MAX_DISTANCE:
                min_class_emo = "Unknown Emotion"

            cv2.putText(frame, f"{min_class} - {min_class_emo}", (x + 20, y + h + 45), cv2.FONT_HERSHEY_PLAIN, 2,
            (0, 255, 0), 2)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return frame
