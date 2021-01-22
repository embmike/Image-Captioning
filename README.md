# Image Captioning
This project is part of my computer vision course of[ Udacity](https://www.udacity.com/course/computer-vision-nanodegree--nd891). The model recognizes objects in images and creates captions for them. The model consists of a Covolutional Neural Network (CNN: Encoder) and a Long Short-Term Memory Network (LSTM: Decoder). The encoder recognizes objects in images and outputs image features to the decoder. The decoder creates a caption based on the image features and its memory.

The encoder is a pre-trained Resnet-152. The classification layer has been replaced by a new feature layer. Both networks use the COCO image database for training. 

[PyTorch](https://pytorch.org/), the open source machine learning framework, is used to implement, train and test the convolutional neural network.

[Restnet-152](https://www.kaggle.com/pytorch/resnet152) is a pre-trained neural convolution network to recognize objects in images. The network has an error rate of 3.57% and is therefore better than a human.

[COCO](https://cocodataset.org/#home), The Microsoft Common Objects in COntext (MS COCO) dataset is a large-scale dataset for scene understanding. The dataset is commonly used to train and benchmark object detection, segmentation, and captioning algorithms.

![Image Captioning CNN-RNN model](images/encoder-decoder.png)
    
    
### Examples

+ **A herd of sheeps**   
  ![Cascade Classifiers Images](/images-with-caption/a-herd-of-sheeps.PNG "A herd of sheeps")   

+ **A plane on the runway**   
  ![CNN Images](/images-with-caption/a-plane-on-the-runway.PNG "A plane on the runway")
   
   
## Important files
- **0_Dataset.ipynb** : Explore this dataset, in preparation for the project.
- **1_Preliminaries.ipynb** : Learn how to load and pre-process data from the COCO dataset.
- **data_loader_val.py** : The Data loader.
- **2_Training.ipynb** : Load, train and validate the convolutional neural network.
- **models.py** : Train the CNN-RNN model.
- **3_Facial_Keypoint_Detection_Complete_Pipeline.ipynb** : Use the trained model to generate captions for images.
    
    
## Installation and usage
Clone the repository
```sh
$ cd <your workspace folder>
$ https://github.com/embmike/Image-Captioning.git
```

You can use the code for example on your computer with [Anaconda](https://www.anaconda.com/) or via cloud computing with [Google Colaboratory](https://colab.research.google.com/). **To train the model you need a GPU.**
    
    
## Licence
This project is licensed under the terms of the [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
