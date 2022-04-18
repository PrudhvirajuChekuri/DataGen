from flask import Flask,render_template,Response,request
import cv2
import tensorflow as tf
import numpy as np
import os
from utils import draw_bounding_box
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
model = tf.keras.models.load_model("Models/90_83")
detect_fn = tf.saved_model.load("Models/my_models/saved_model")
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def detectandupdate(img):
    path = "static/" + str(img)
    image = cv2.imread(path)
    coordinates = draw_bounding_box(image, detect_fn)
    
    for (y, h, x, w) in coordinates:
        cv2.rectangle(image,(x,y),(w, h),(0, 255, 0),2)
        img2 = image[y:h, x:w]
        img2 = tf.image.resize(img2, size = [128, 128])
        pred = model.predict(tf.expand_dims(img2, axis=0))
        pred_class = class_names[tf.argmax(pred, axis = 1).numpy()[0]]
        if x > 20 and y > 40:
            cv2.putText(image, pred_class, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(image, pred_class, (x + 10, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    path2 = f"static/pred{img}"
    cv2.imwrite(path2, image)


    return ([img, "pred" + img])


def picdelete(path):
    os.rmdir(path)