import tensorflow as tf
import numpy as np
from tensorflow import keras

mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape, y_train.shape)

# normalize: 0,255 -> 0,1
x_train, x_test = x_train / 255.0, x_test / 255.0

# model
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10),
    keras.layers.Softmax()
])

# loss and optimizer
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optim = keras.optimizers.Adam(lr=0.001)
metrics = ["accuracy"]

model.compile(loss=loss, optimizer=optim, metrics=metrics)

# training
batch_size = 64
epochs = 150

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=1)

print("Evaluate:")
model.evaluate(x_test, y_test, batch_size=batch_size, verbose=2)


#save the model
model.save("neural_network.h5")

#print("\nModel avaluaten from the saved model")
#new_model = keras.models.load_model("nn2.h5")
#new_model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)

#scores = new_model.evaluate(x_test, y_test)
#print("Accuracy: ", scores[1]*100)
