# Overview
This code uses the Keras deep learning library and TensorFlow backend to build a convolutional neural network (CNN) to classify German traffic signs using the German Traffic Sign Recognition Benchmark dataset.

# Libraries used
- Pandas
- Numpy
- Matplotlib
- TensorFlow
- Dataset
The dataset from `kaggle` is divided into two parts: train and test datasets. Both datasets are read into pandas dataframes. The train dataset has 39209 images and the test dataset has 12630 images. Each image in the dataset contains metadata like width, height, and classid.

# Exploring the dataset
The head of both the train and test dataframes is printed to the console. Additionally, the info() method is used to display the number of rows, non-null counts, and datatypes of each column in both datasets.

# Creating the 43 class names
The 43 class names are created as a dictionary with keys as the ClassId and values as the name of the traffic sign.

# Building the model
The model uses the Tiny VGG architecture. The model consists of four convolutional layers with `max-pooling` and dropout layers, followed by a fully connected layer and output layer with `softmax` activation. The model is compiled with the `Adam` optimizer and `categorical cross-entropy` loss function.

# Model training and evaluation
The model is trained on the train dataset and evaluated on the test dataset using accuracy as the evaluation metric. The trained model is then used to predict the class labels of images in the test dataset. The classification report is generated to display precision, recall, and F1-score for each class label. The confusion matrix is also generated to display the number of correct and incorrect predictions for each class.

## Accuracy of 95%
the accuracy for 10 epochs is about 95%
