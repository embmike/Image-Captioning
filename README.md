# Facial Keypoint Detection
This project is part of my computer vision course of[ Udacity](https://www.udacity.com/course/computer-vision-nanodegree--nd891). A neural convolution network recognizes faces in images based on learned facial keypoints. 
These keypoints mark important areas of the face: the eyes, corners of the mouth, the nose, etc. These keypoints are relevant for a variety of tasks, such as face filters, emotion recognition, pose recognition, and so on. 

Likewise the computer vision method hair feature-based cascade uses these features, see [OpenCV-Python tutorial](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html).

[PyTorch](https://pytorch.org/), the open source machine learning framework, is used to implement, train and test the convolutional neural network.
    
    
### Examples

+ **Detect all faces using Haar Cascade Classifiers using OpenCV**
![Cascade Classifiers Images](/output_images/cascade_classifier.png "Detect all faces using Haar Cascade Classifiers using OpenCV")   

+ **Detect facial keypoint with a Convolutional Neural Network**
![CNN Images](/output_images/cnn_persons.png "Detect facial keypoint with a Convolutional Neural Network")
   
   
## Important files
- **1_Load_and_Visualize_Data.ipynb** : Learn to create a data loader and visualize the data
- **data_load.py** : The Data loader
- **2_Define_the_Network_Architecture.ipynb** : Load, train and validate the convolutional neural network.
- **models.py** : The Convolutional neural network
- **3_Facial_Keypoint_Detection_Complete_Pipeline.ipynb** : Test and compare the model with the cascade classifier.
    
    
## Installation and usage
Clone the repository
```sh
$ cd <your workspace folder>
$ git clone https://github.com/embmike/Facial-Keypoint-Detection.git
```

You can use the code for example on your computer with [Anaconda](https://www.anaconda.com/) or via cloud computing with [Google Colaboratory](https://colab.research.google.com/)
    
    
## Licence
This project is licensed under the terms of the [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
