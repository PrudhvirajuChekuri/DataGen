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

def MakeZipFile(filepath):
    shutil.make_archive('./dataset','zip',filepath)

def MakeZipLabel(filepath):
    shutil.make_archive('./labeldata','zip',filepath)

def MakeZipFed(filepath):
    shutil.make_archive('./feddata', 'zip', filepath)

#Below functions are used to delete the input and output files after the user exits.
@app.route('/deletelabel')
def deletelabel():
    if "labeldata.zip" in os.listdir("./"):
        os.remove("labeldata.zip")
    return ("nothing")

@app.route('/deletedata')
def deletedata():
    if "dataset.zip" in os.listdir("./"):
        os.remove("dataset.zip")
    return ("nothing")

@app.route('/deletefed')
def deletefed():
    if "feddata.zip" in os.listdir("./"):
        os.remove("feddata.zip")
    return ("nothing")

@app.route('/')
def home():
	return render_template("home.html")

def allowed_file(filename):
    return ('.' in filename and
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS)

@app.route('/bulkdetect', methods=['GET', 'POST'])
def bulkdetect():
    dirs = ["preparedataset","preparedataset/Angry", "preparedataset/Disgust", "preparedataset/Fear", 
    "preparedataset/Happy", "preparedataset/input", "preparedataset/Neutral", "preparedataset/Sad",
    "preparedataset/Surprise"]

    if dirs[0] in os.listdir("./"):
        shutil.rmtree("preparedataset")#If the directory already exits then deletes it.

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

    path='dataset.zip'
    shutil.rmtree("preparedataset")
    return send_file(path,as_attachment=True)

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
                height, width, depth = image.shape
                coordinates = draw_bounding_box(image, detect_fn)#Get the bboxes.

                #Since the file to be generated is xml and follows a format as shown in README file.
                #First creating a string that consists the head part of the file.
                content = f"<annotation>\n\t<folder>Some_Folder</folder>\n\t<filename>{filename}</filename>\n\t<path>Some_path</path>\n\t<source>\n\t\t<database>Unknown</database>\n\t</source>\n\t<size>\n\t\t<width>{width}</width>\n\t\t<height>{height}</height>\n\t\t<depth>{depth}</depth>\n\t</size>\n\t<segmented>0</segmented>"

                #Loop over bboxes and conactenate the objects(bboxes) information to the content string.
                for (y, h, x, w) in coordinates:
                    cv2.rectangle(image,(x,y),(w, h),(0, 255, 0),2)
                    content += f"\n\t<object>\n\t\t<name>face</name>\n\t\t<pose>Unspecified</pose>\n\t\t<truncated>0</truncated>\n\t\t<difficult>0</difficult>\n\t\t<bndbox>\n\t\t\t<xmin>{x}</xmin>\n\t\t\t<ymin>{y}</ymin>\n\t\t\t<xmax>{w}</xmax>\n\t\t\t<ymax>{h}</ymax>\n\t\t</bndbox>\n\t</object>"

                content += "\n</annotation>"#Cat the tail. This flow is followed to make it work for multiple bboxes.
                filepath="labeleddata/output/"+ str(filename)[:-4]+'.xml'#Save it as an xml file.
                with open(filepath, 'a') as f:
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

                height, width, depth = image.shape
                content = f"<annotation>\n\t<folder>Some_Folder</folder>\n\t<filename>{filename}</filename>\n\t<path>Some_path</path>\n\t<source>\n\t\t<database>Unknown</database>\n\t</source>\n\t<size>\n\t\t<width>{width}</width>\n\t\t<height>{height}</height>\n\t\t<depth>{depth}</depth>\n\t</size>\n\t<segmented>0</segmented>"

                for (y, h, x, w) in coordinates:
                    cv2.rectangle(image,(x,y),(w, h),(0, 255, 0),2)
                    img = image[y:h, x:w]
                    img = tf.image.resize(img, size = [128, 128])
                    pred = model.predict(tf.expand_dims(img, axis=0))
                    pred_class = class_names[tf.argmax(pred, axis = 1).numpy()[0]]
                    #The only thing that differs from the above one is replacing the face with {pred_class} as below.
                    content += f"\n\t<object>\n\t\t<name>{pred_class}</name>\n\t\t<pose>Unspecified</pose>\n\t\t<truncated>0</truncated>\n\t\t<difficult>0</difficult>\n\t\t<bndbox>\n\t\t\t<xmin>{x}</xmin>\n\t\t\t<ymin>{y}</ymin>\n\t\t\t<xmax>{w}</xmax>\n\t\t\t<ymax>{h}</ymax>\n\t\t</bndbox>\n\t</object>"

                content += "\n</annotation>"
                filepath="fed/output/"+ str(filename)[:-4]+'.xml'
                with open(filepath, 'a') as f:
                        f.write(content)

        MakeZipFed('./fed')
    return render_template('donefed.html')

@app.route('/fed')
def fed():
    return render_template('inputforfed.html')

@app.route('/getinlab')
def getinlab():
    return render_template('inputforbound.html')

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

if __name__=="__main__":
    app.run(debug=True)