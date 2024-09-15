import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tensorflow.keras import datasets, layers, models

matplotlib.use('Agg')

# Load CIFAR-10 dataset
(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()

# Normalize images
training_images, testing_images = training_images / 255.0, testing_images / 255.0

# Class names for the CIFAR-10 dataset
class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Plot the first 16 images
plt.figure(figsize=(10, 10))  # larger figure for better visibility

for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(training_images[i])  # , cmap=plt.cm.binary | Add cmap for better contrast, remove it if not needed
    plt.xticks([])  # Remove x-axis ticks
    plt.yticks([])  # Remove y-axis ticks
    plt.title(class_names[training_labels[i][0]])

plt.tight_layout()  # Prevent overlap of subplots
plt.show()

# plt.savefig('output_image.png')