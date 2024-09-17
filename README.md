# Image Classifier using CIFAR-10 and TensorFlow/Keras

This project is an image classification model built using the CIFAR-10 dataset. It leverages the TensorFlow/Keras library to train a Convolutional Neural Network (CNN) for recognizing images from ten different classes, including planes, cars, birds, cats, and more. The project demonstrates how to preprocess data, train a model, and make predictions on new images.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Training the Model](#training-the-model)
- [Making Predictions](#making-predictions)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

The goal of this project is to create a basic image classifier using the CIFAR-10 dataset. The CIFAR-10 dataset contains 60,000 32x32 color images in 10 classes, with 6,000 images per class. The model uses a convolutional neural network (CNN) for classifying the images into one of the ten categories.

The project includes the following main steps:

1. Loading and normalizing the CIFAR-10 dataset.
2. Visualizing a subset of the training data.
3. Training a CNN on the training dataset.
4. Evaluating the model's performance on the test dataset.
5. Using the trained model to make predictions on new images.

## Features

- Preprocesses CIFAR-10 images and labels.
- Visualizes sample images from the dataset.
- CNN model with three convolutional layers followed by dense layers for classification.
- Model evaluation and prediction on new images.

## Dataset

The project uses the CIFAR-10 dataset, which contains 60,000 color images in 10 different categories:

- Airplane
- Car
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

The dataset is automatically loaded from TensorFlow’s `keras.datasets` module.

## Installation

To run this project, ensure you have Python 3.x installed, along with the following libraries:

- TensorFlow/Keras
- OpenCV
- NumPy
- Matplotlib

You can install the required dependencies using `pip`:

```bash
pip install tensorflow opencv-python-headless numpy matplotlib
```

## Usage

To use this project, follow these steps:

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/rufilboss/image-classifier.git
   cd image-classifier
   ```

2. (Optional) Train the model:
   If you want to retrain the model with custom settings, uncomment the training section in the code.

3. Run the image classifier:

   ```bash
   python3 main.py
   ```

4. Make predictions on new images by placing an image named `test.jpg` in the project directory.

## Training the Model

The model is built using TensorFlow/Keras and consists of the following layers:

- 3 Convolutional layers followed by MaxPooling layers
- 1 Flatten layer to convert the 2D output to 1D
- 2 Dense (fully connected) layers, with the last layer being the output layer with 10 units (one for each class) and a softmax activation function.

The model is trained for 10 epochs using the Adam optimizer and sparse categorical cross-entropy as the loss function.

To train the model, uncomment the training code in `main.py`:

```python
# model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images, testing_labels))
```

After training, the model is saved to a file `image_classifier.keras`.

## Making Predictions

Once the model is trained or loaded, you can make predictions on new images. The project demonstrates this by loading an image, resizing it to the required dimensions (32x32), and predicting its class:

```python
img = cv.imread('test.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img = resize(img, (32, 32))
img = img_to_array(img)
prediction = model.predict(np.array([img]) / 255)
index = np.argmax(prediction)
print(f"Prediction is: {class_names[index]}")
```

This code assumes that the image is named `test.jpg` and is located in the project directory.
You can replace `test.jpg` with any image of your choice.

## Results

The model has a moderate accuracy on the CIFAR-10 test set. With additional tuning or training for more epochs, accuracy could be improved.

Example prediction:

```sh
Prediction is: Horse
```

## Contributing

Contributions to this project are welcome! Feel free to open issues or submit pull requests to enhance the project.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

```sh
You can modify and customize this `README.md` to fit additional details or project-specific configurations you want to highlight.
```
