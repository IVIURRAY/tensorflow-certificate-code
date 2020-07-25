"""
Video: bit.ly/tfw-lab4

In the previous lessons you saw how to do fashion recognition using a Deep Neural Network
(DNN) containing three layers -- the input layer (in the shape of the data),
the output layer (in the shape of the desired output) and a hidden layer.
You experimented with the impact of different sized of hidden layer,
number of training epochs etc on the final accuracy.
"""

import tensorflow as tf
print(tf.__version__)

# Load the dataset
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# 60,000 images reduced to 28x28 with 1 colour depth.
training_images=training_images.reshape(60000, 28, 28, 1)
training_images=training_images / 255.0
test_images = test_images.reshape(10000, 28, 28, 1)
test_images=test_images/255.0

# Train and test the data set.
model = tf.keras.models.Sequential([

    # Here we're applying convolution abstract features from the images and pooling
    # to reduce the image size on processing.
    # The 64 are the number of convolutions to generate.
    # The (3, 3) is the size of our filters and (2, 2) half-ing our image size.
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    # This is the same as before
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')

])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(training_images, training_labels, epochs=5)
test_loss = model.evaluate(test_images, test_labels)

