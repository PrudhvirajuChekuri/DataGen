from flask import Flask,render_template,Response,request
import cv2
import tensorflow as tf
import numpy as np
from utils import draw_bounding_box
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
model = tf.keras.models.load_model("Models/FEC")
detect_fn = tf.saved_model.load("Models/FaceDetector/saved_model")
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def pushIntoFile(img):
    path = "./preparedataset/input/" + str(img)
    image = cv2.imread(path)
    coordinates = draw_bounding_box(image, detect_fn)
    for (y, h, x, w) in coordinates:
        img2 = image[y:h, x:w]
        img2 = tf.image.resize(img2, size = [128, 128])
        pred = model.predict(tf.expand_dims(img2, axis=0))
        pred_class = class_names[tf.argmax(pred, axis = 1).numpy()[0]]

        path = "preparedataset/"+ pred_class+"/"+ str(img)
        cv2.imwrite(path, image)


    return ([img, "pred" + img])