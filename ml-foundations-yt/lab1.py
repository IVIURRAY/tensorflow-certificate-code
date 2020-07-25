"""
YouTube Vidoe: https://www.youtube.com/watch?v=_Z9TRANg4c0&list=PLOU2XLYxmsII9mzQ-Xxug4l2o04JBrkLV&index=2&t=0s
Colab Workbook: https://colab.research.google.com/github/lmoroney/mlday-tokyo/blob/master/Lab1-Hello-ML-World.ipynb


Introduction
We've two sequences of numbers and we want to use ML to work out
the relationship between them.

A: [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
B: [-2.0, 1.0, 4.0, 7.0, 10.0, 13.0]

The relationship is 3x + 1. But we want to see if we can make a
neural net to work this out for us.
"""

import tensorflow as tf
import numpy as np
from tensorflow import keras

# Create a model of one later and one neuron.
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

# Create the model
model.compile(optimizer='sgd', loss='mean_squared_error')

# Provide data to model
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)

# Train the Neural Network (over 500 epochs)
model.fit(xs, ys, epochs=500)

# Use the model to predict some output
print(model.predict([10.0]))

"""
...
1/1 [==============================] - 0s 270us/step - loss: 5.6018e-07
[[30.997814]]

Notice how the value we predicted is not 31 but 30.99.
This is because ML works on probability and we don't have data out to there.
So the model is doing its best to predict where this value lies but it can't be for sure.
"""


####################################

