import tensorflow as tf
from tensorflow import keras


mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
#print(x_train.shape, y_train.shape)

# normalize: 0,255 -> 0,1
x_train, x_test = x_train / 255.0, x_test / 255.0

model = keras.models.load_model("./neural_network.h5")

score = model.evaluate(x_test, y_test)

#print(score)
print(score[1]*100)