from flask import Flask, render_template, Response, request
import cv2
import tensorflow as tf
import numpy as np
import os
import shutil
from detectandstore import detectandupdate
from pushfile import pushIntoFile
from werkzeug.utils import secure_filename
from flask import current_app
from flask import send_file
from getmail import send_mail
from utils import draw_bounding_box
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
app.config['DEBUG'] = True

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
model = tf.keras.models.load_model("Models/FEC")#Load the facial emotion classifier.
detect_fn = tf.saved_model.load("Models/FaceDetector/saved_model")#Load the face detector.

class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
static_files = ['Aboutb.jpg', 'cam.jpg', 'Classifyb.jpg', 'Classifydoneb.jpg', 'Detectb.jpg', 'Detectdoneb.jpg',
                'display.css', 'eye.png', 'eye1.png', 'feedbackb.jpg', 'Homeb.jpg', 'loading-page.gif',
                'Picdetectb.jpg', 'Picuploadb.jpg', 'thumbsup.jpg', '1.jpg', '3.jpg']

def MakeZipFile(filepath):
    shutil.make_archive('./dataset','zip',filepath)

def MakeZipLabel(filepath):
    shutil.make_archive('./labeldata','zip',filepath)

def MakeZipFed(filepath):
    shutil.make_archive('./feddata', 'zip', filepath)

@app.route('/picdelete')
def picdelete():
    #When this function is called all the files that are not present in the
    #list static_files will be deleted.
    for file in os.listdir("static"):
        if file not in static_files:
            os.remove(f"static/{file}")
    return ("nothing")

#Below functions are used to delete the input and output files after the user exits.
@app.route('/deletelabel')
def deletelabel():
    os.remove("labeldata.zip")
    return ("nothing")

@app.route('/deletedata')
def deletedata():
    os.remove("dataset.zip")
    return ("nothing")

@app.route('/deletefed')
def deletefed():
    os.remove("feddata.zip")
    return ("nothing")

@app.route('/')
def home():
	return render_template("home.html")

def allowed_file(filename):
    return ('.' in filename and
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS)

@app.route('/detectpic', methods=['GET', 'POST'])
def detectpic():
    UPLOAD_FOLDER = 'static'
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    if request.method == 'POST':

        file = request.files['file']

        if file and allowed_file(file.filename):

            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            result =detectandupdate(filename)
            return render_template('showdetect.html', orig=result[0], pred=result[1])

@app.route('/picdetect')
def picdetect():
    return render_template('picdetect.html')

@app.route('/bulkdetect', methods=['GET', 'POST'])
def bulkdetect():
    dirs = ["preparedataset","preparedataset/Angry", "preparedataset/Disgust", "preparedataset/Fear", 
    "preparedataset/Happy", "preparedataset/input", "preparedataset/Neutral", "preparedataset/Sad",
    "preparedataset/Surprise"]

    if dirs[0] in os.listdir("/home/ubuntu/flaskapp/"):
        shutil.rmtree("preparedataset")#If the directory exits then deletes it.

    for dir in dirs:
        os.mkdir(dir)
        
    UPLOAD_FOLDER = './preparedataset/input'
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    if request.method == 'POST':
        for file in request.files.getlist('file'):
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                pushIntoFile(filename)
        MakeZipFile('./preparedataset')
        return render_template('donedow.html')

@app.route('/red_to_bulkin')
def red_to_bulkin():
    return render_template('bulkinput.html')

@app.route('/makebound', methods=['GET', 'POST'])
def makebound():
    dirs = ["labeleddata", "labeleddata/input", "labeleddata/output"]

    if dirs[0] in os.listdir("/home/ubuntu/flaskapp/"):
        shutil.rmtree("labeleddata")

    for dir in dirs:
        os.mkdir(dir)
    UPLOAD_FOLDER = './labeleddata/input'
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    if(request.method == 'POST'):
        for file in request.files.getlist('file'):
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)#Get the filename.
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))#Save it.
                path = "labeleddata/input/" + str(filename)
                image = cv2.imread(path)#Read it using cv2.
                coordinates = draw_bounding_box(image, detect_fn)#Get the bboxes.

                #Loop over bboxes.
                for (y, h, x, w) in coordinates:
                    cv2.rectangle(image,(x,y),(w, h),(0, 255, 0),2)
                    filepath="labeleddata/output/"+ str(filename)[:-4]+'.txt'
                    with open(filepath, 'a') as f:
                        content=str(x) + ' ' + str(y) + ' ' + str(w) + ' '+ str(h) + '\n'
                        f.write(content)
        MakeZipLabel('./labeleddata')
    return render_template('donelabel.html')

@app.route('/fedbound', methods=['GET', 'POST'])
def fedbound():
    dirs = ["fed", "fed/input", "fed/output"]

    if dirs[0] in os.listdir("/home/ubuntu/flaskapp/"):
        shutil.rmtree("fed")

    for dir in dirs:
        os.mkdir(dir)

    UPLOAD_FOLDER = './fed/input'
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    if(request.method == 'POST'):
        for file in request.files.getlist('file'):
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                path = "fed/input/" + str(filename)
                image = cv2.imread(path)
                coordinates = draw_bounding_box(image, detect_fn)

                for (y, h, x, w) in coordinates:
                    cv2.rectangle(image,(x,y),(w, h),(0, 255, 0),2)
                    img = image[y:h, x:w]
                    img = tf.image.resize(img, size = [128, 128])
                    pred = model.predict(tf.expand_dims(img, axis=0))
                    pred_class = class_names[tf.argmax(pred, axis = 1).numpy()[0]]
                    filepath="fed/output/"+ str(filename)[:-4]+'.txt'
                    with open(filepath, 'a') as f:
                        content=str(x) + ' ' + str(y) + ' ' + str(w) + ' '+ str(h) + ', ' + pred_class + '\n'
                        f.write(content)
        MakeZipFed('./fed')
    return render_template('donefed.html')

@app.route('/fed')
def fed():
    return render_template('inputforfed.html')

@app.route('/getinlab')
def getinlab():
    return render_template('inputforbound.html')

#Below three functions are used to download the generated zip files.
@app.route('/downloadDS', methods=['GET', 'POST'])
def downloaddataset():
    path='dataset.zip'
    shutil.rmtree("preparedataset")
    return send_file(path,as_attachment=True)

@app.route('/downloadLB', methods=['GET', 'POST'])
def downloadLabel():
    path='labeldata.zip'
    shutil.rmtree("labeleddata")
    return send_file(path,as_attachment=True)

@app.route('/download', methods=['GET', 'POST'])
def downloadfeddataset():
    path='feddata.zip'
    shutil.rmtree("fed")
    return send_file(path,as_attachment=True)

@app.route('/feedback')
def feedback():
    return render_template('feedback.html')

@app.route('/sentsafe',methods=['GET', 'POST'])
def send_sentsafe():
    if request.method == 'POST':
        email = request.form['email']
        comments = request.form['comments']
        name=request.form['name']
        comments=email+"  \n "+name+"  \n "+comments
        send_mail(email,comments)
    return render_template('sentfeed.html')
