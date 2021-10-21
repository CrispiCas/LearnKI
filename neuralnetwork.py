import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
#print(x_train.shape, y_train.shape)

# normalize: 0,255 -> 0,1
x_train, x_test = x_train / 255.0, x_test / 255.0

# model
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10),
])

#print(model.summary())

# another way to build the Sequential model:
#model = keras.models.Sequential()
#model.add(keras.layers.Flatten(input_shape=(28,28))
#model.add(keras.layers.Dense(128, activation='relu'))
#model.add(keras.layers.Dense(10))

# loss and optimizer
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optim = keras.optimizers.Adam(lr=0.001)
metrics = ["accuracy"]

model.compile(loss=loss, optimizer=optim, metrics=metrics)

# training
batch_size = 64
epochs = 5

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=1)

#evaluate

model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)

#prediction
probability_model = keras.models.Sequential([
    model,
    keras.layers.Softmax()
])

prediction = probability_model(x_test)
pred0 = prediction[0]
print(pred0)
label0 = np.argmax(pred0)
print(label0)

#model + softmax

prediction = model(x_test)
prediction = tf.nn.softmax(prediction)
print(label0)


pred05 = prediction[0:5]
print(pred05.shape)
label05 = np.argmax(pred05, axis=1)
print(label05)