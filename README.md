# FashionMNIST_Classification
This project focuses on the application of machine learning techniques in the fashion industry using the Fashion MNIST dataset, which comprises 60,000 training and 10,000 test images of various clothing items at 28x28 pixels. Two algorithms, Convolutional Neural Network (CNN) and Multilayer Perceptron (MLP), are built and compared to classify these fashion items into ten categories. The CNN model, known for handling spatial data through convolutional layers, serves as the primary algorithm, while the MLP model provides a comparative baseline. This repository includes all relevant code files, datasets, and documentation needed to reproduce and understand the model's training and application.


**CNNClassifier_FashionMNIST.py:**

    **1. Libraries Used:** Numpy, Pandas, Matplotlib, Seaborn, TensorFlow Keras
 
    **2. Purpose:** Implements a Convolutional Neural Network (CNN) to classify images from the Fashion MNIST dataset. The model consists of convolutional
    layers for feature extraction and dense layers for classification. The script handles data loading, preprocessing, model training, and evaluation, 
    displaying the confusion matrix and classification report.

  
**MLPClassifier_FashionMNIST.py:**

  **Libraries Used:** Numpy, Pandas, Matplotlib, Seaborn, TensorFlow Keras
  
  **Purpose:** Implements a Multilayer Perceptron (MLP) to serve as a baseline comparison for the CNN model. It processes images from the Fashion MNIST dataset
  by flattening them into vectors and classifying them using dense layers. This file covers the model's setup, training, performance evaluation, and
  visualization of results.
  

**RelaLifeApplication_CNN.py**

  **Libraries Used:** PIL, Matplotlib, Numpy, TensorFlow Keras
  
  **Purpose:** Demonstrates a real-life application of the trained CNN model to classify clothing items in a retail setting. This script processes real-world
  images, adapts them to the input format of the CNN, and visualizes predictions next to original images to showcase practical deployment capabilities.
