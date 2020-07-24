import os
import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image

facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = model_from_json(open("fer.json", "r").read())
#load weights
model.load_weights('fer.h5')
#font = cv2.FONT_HERSHEY_SIMPLEX

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    # returns camera frames along with bounding boxes and predictions
    def get_frame(self):
        _, fr = self.video.read()
        gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        faces_detected = facec.detectMultiScale(gray_fr, 1.3, 5)

        for (x, y, w, h) in faces_detected:
            cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0),thickness=7)
            roi_gray=gray_fr[y:y+w,x:x+h]
            #cropping region of interest i.e. face area from  image
            roi_gray=cv2.resize(roi_gray,(48,48))
            img_pixels = image.img_to_array(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis = 0)
            img_pixels /= 255

            predictions = model.predict(img_pixels)

            #find max indexed array
            max_index = np.argmax(predictions[0])

            emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
            predicted_emotion = emotions[max_index]

            cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        '''resized_img = cv2.resize(fr, (1000, 700))
        cv2.imshow('Facial emotion analysis ',resized_img)'''

        _, jpeg = cv2.imencode('.jpg', fr)
        return jpeg.tobytes()

