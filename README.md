# DataGen(Data Annotation Tool)📁

If you are here and this is your first reading of mine, you can go [here](https://github.com/PrudhvirajuChekuri/EmoViz/tree/master) and check that repo first.

Jump over to the Installation section using the table of contents if you don't want to know much about the project. 

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#Description">Description</a></li>
    <li><a href="#About the Models">About the Models</a></li>
    <li>
      <a href="#How the Output looks">How the Output looks</a>
      <ul>
        <li><a href="#Facial Emotion Classification">Facial Emotion Classification</a></li>
      </ul>
      <ul>
        <li><a href="#Face Detection">Face Detection</a></li>
      </ul>
      <ul>
        <li><a href="#Facial Emotion Detection">Facial Emotion Detection</a></li>
      </ul>
    </li>
    <li>
      <a href="#Installation">Installation</a>
      <ul>
        <li><a href="#Requirements">Requirements</a></li>
        <li><a href="#Setup">Setup</a></li>
      </ul>
    </li>
    <li><a href="#Inference">Inference</a></li>
    <li><a href="#License">License</a></li>
    <li><a href="#Credits">Credits</a></li>
  </ol>
</details>

# 📝Description

When I'm building the facial emotion detector data annotation and organization became a pain for me. So after completed that project I decided to bulid this data annotation tool. This will help the folks who are intrested in building models related to face like face detection, facial emotion classification, facial emotion detection etc.

<p align="right">(<a href="#top">back to top</a>)</p>

# 🤖About the Models

If you haven't check this [repo](https://github.com/PrudhvirajuChekuri/EmoViz/tree/master) yet, go there and check it first to know more about the models used.

<p align="right">(<a href="#top">back to top</a>)</p>

# 📂How the Output looks

## Facial Emotion Classification
After giving the input, the dataset will be generated and you can download it as z .zip file.
The dataset.zip file looks like this

![image](https://user-images.githubusercontent.com/96725900/164470429-5d47e46a-a0e8-4a80-be75-7a8e5159e2c4.png)

with all the images in their respective directories.

## Face detection
A .zip file will be generated containing all the bounding boxes of faces in xml format which is the required format for TensorFlow 2.0 object detection.
Ths labeldata.zip looks like this

![image](https://user-images.githubusercontent.com/96725900/164471846-d7dcc240-ea3f-4b00-bb33-677b8e8c01bc.png)

The output directory contains all the outputs for the given images.
The output xml file looks like this

![image](https://user-images.githubusercontent.com/96725900/164473595-40631524-7e57-4ada-9e20-2d7168731eb4.png)

same as the output that is generated by the labelImg software.

## Facial Emotion Detection
It looks same as the above option but there is only one change i.e change in the name part of the xml files. The above files consist of name face for all the boxes and these files consist of the respective emotion of the bounding boxes.

# 🖥Installation

## 🛠Requirements

* Python 3.7+ 
* Other requirement are in the requirements.txt and can be installed as shown below.

## ⚙️Setup
```
pip install -r requirements.txt
```
That's it you're set to go.

<p align="right">(<a href="#top">back to top</a>)</p>

# 🎯Inference
```
python app.py
```
<p align="right">(<a href="#top">back to top</a>)</p>

# ⚖License
Distributed under the MIT License. See LICENSE.txt for more information.

<p align="right">(<a href="#top">back to top</a>)</p>

# Credits
* http://www.whdeng.cn/raf/li_RAFDB_2017_CVPR.pdf

<p align="right">(<a href="#top">back to top</a>)</p>