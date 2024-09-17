import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.image import resize
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense


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

# TRAINING THE MODEL

# 
training_images = training_images[:20000]
training_labels = training_labels[:20000]
testing_images = testing_images[:4000]
testing_labels = testing_labels[:4000]

# -----------------------------------------------------UNCOMMENT THIS CODE TO RETRAIN THE DATA------------------------------------------------------------------
'''
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# Compling
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training
model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images, testing_labels))

# Evaluate the model
loss, accuracy = model.evaluate(testing_images, testing_labels)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

# Save the model
model.save('image_classifier.keras')
'''

model = models.load_model('image_classifier.keras')

img = cv.imread('test.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# Resizing the image to fit the model's expected input
img = resize(img, (32, 32))  # Resize to (32, 32, 3), adjust based on our model
img = img_to_array(img)

plt.imshow(img, cmap='binary')

prediction = model.predict(np.array([img]) / 255)
index = np.argmax(prediction)
print(f"Prediction is: {class_names[index]}")
