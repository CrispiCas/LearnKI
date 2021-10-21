import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random

mnist = keras.datasets.mnist

model = keras.models.load_model("./neural_network.h5")


(x_train, y_train), (x_test, y_test) = mnist.load_data()
#print(x_train.shape, y_train.shape)

# normalize: 0,255 -> 0,1
x_train, x_test = x_train / 255.0, x_test / 255.0

img_width, img_height = 28, 28

def test1():
    use_samples = [5, 38, 3939, 27389]
    samples_to_predict = []

    # Generate plots for samples
    for sample in use_samples:
        # Generate a plot
        reshaped_image = x_train[sample].reshape((img_width, img_height))
        plt.imshow(reshaped_image)
        plt.show()
        # Add sample to array for prediction
        samples_to_predict.append(x_train[sample])

    samples_to_predict = np.array(samples_to_predict)
    #print(samples_to_predict.shape)


    predictions = model.predict(samples_to_predict)
    #print(predictions)

    classes = np.argmax(predictions, axis = 1)
    print(classes)

def test2():
    rand = random.randint(0,500)
    sample = x_test[rand]
    samples_to_predict = []

    # Generate plots for samples
    # Generate a plot
    reshaped_image = sample.reshape((img_width, img_height))
    plt.imshow(reshaped_image)
    plt.show()
    # Add sample to array for prediction
    samples_to_predict.append(sample)

    samples_to_predict = np.array(samples_to_predict)
    #print(samples_to_predict.shape)


    predictions = model.predict(samples_to_predict)
    #print(predictions)

    classes = np.argmax(predictions, axis = 1)
    print("The Number is: ")
    print(classes)


test2()