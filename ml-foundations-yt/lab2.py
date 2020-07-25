"""
Link: bit.ly/tfw-lab2cv

IMAGE CLASSIFICATION

We want to predict some images this time.
Given the MNIST data set of images of clothing.
Create a neural network that can predict what the image is.
"""

import matplotlib.pyplot as plt
import tensorflow as tf
print(tf.__version__)

# Split the data into training and test data sets.
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# # Show a image just to see
plt.imshow(training_images[0])
print(training_labels[0])
print(training_images[0])

# Normalise the images so they're all the same
training_images = training_images / 255.0
test_images = test_images / 255.0

# Create the Model to Run
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    # We want to out 10 as we've 10 different labels and clothing types
    # Softmax picks the most likely value from the 10 neurons and sets it to 1 and the rest to 0.
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

model.fit(training_images, training_labels, epochs=5)

"""
80% accuracy at the start - 0.8261 

Epoch 1/5
1875/1875 [==============================] - 2s 1ms/step - loss: 0.4947 - accuracy: 0.8261
Epoch 2/5
1875/1875 [==============================] - 2s 1ms/step - loss: 0.3709 - accuracy: 0.8657
Epoch 3/5
1875/1875 [==============================] - 3s 1ms/step - loss: 0.3331 - accuracy: 0.8777
Epoch 4/5
1875/1875 [==============================] - 2s 1ms/step - loss: 0.3106 - accuracy: 0.8853
Epoch 5/5
1875/1875 [==============================] - 2s 1ms/step - loss: 0.2941 - accuracy: 0.8908

90% accuracy at the end - 0.8908
"""

# Now lets see how it does against our test images.
model.evaluate(test_images, test_labels)

"""
It had a 88% accuracy - 0.8752 

313/313 [==============================] - 0s 935us/step - loss: 0.3487 - accuracy: 0.8752
"""