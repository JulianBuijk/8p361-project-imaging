import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.callbacks import TensorBoard

# Set model option: 1 = no hidden layers, 2 = three hidden layers with ReLU, 3 = three hidden layers with linear activations
model_option = 2 # Change this value to 1, 2, or 3 as needed

# Load dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Split training data into train and validation sets
X_train, X_val = train_test_split(X_train, test_size=0.10, random_state=101)
y_train, y_val = train_test_split(y_train, test_size=0.10, random_state=101)

# Reshape images and normalize to [0, 1]
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255
X_val   = X_val.reshape(-1, 28, 28, 1).astype('float32') / 255
X_test  = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255

# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_val   = to_categorical(y_val, 10)
y_test  = to_categorical(y_test, 10)

# Create the model based on model_option
model = Sequential()
model.add(Flatten(input_shape=(28,28,1)))  # Input layer

if model_option == 1:
    # No hidden layers: input directly connected to output
    model.add(Dense(10, activation='softmax'))
elif model_option == 2:
    # Three hidden layers with ReLU activations
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(10, activation='softmax'))
elif model_option == 3:
    # Three hidden layers with linear activations
    model.add(Dense(128, activation='linear'))
    model.add(Dense(64, activation='linear'))
    model.add(Dense(32, activation='linear'))
    model.add(Dense(10, activation='softmax'))
else:
    print("Invalid model_option value! Please set it to 1, 2, or 3.")
    exit()

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# Set model name based on the chosen option
if model_option == 1:
    model_name = "no_hidden_layer_model"
elif model_option == 2:
    model_name = "three_hidden_layer_relu_model"
elif model_option == 3:
    model_name = "three_hidden_layer_linear_model"

# TensorBoard callback
tensorboard = TensorBoard(log_dir="logs/" + model_name)

# Train the model
model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=1, validation_data=(X_val, y_val), callbacks=[tensorboard])

# Evaluate the model
score = model.evaluate(X_test, y_test, verbose=0)

# Print final loss and accuracy
print("Loss: ", score[0])
print("Accuracy: ", score[1])
