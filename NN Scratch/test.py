import numpy as np
import tensorflow as tf
from archi import Dense, Model
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train[:10000].reshape(10000,784)
x_train = x_train/255
y_train = y_train[:10000]
x_test = x_test[:1000].reshape(1000,784)/255
y_test = y_test[:1000]

ytrain = np.zeros((10000,10))
ytest = np.zeros((1000,10))

for i in range(len(ytrain)):
    ytrain[i,y_train[i]] = 1

for i in range(len(ytest)):
    ytest[i,y_test[i]] = 1


layers = [{'units':512,'activation':'relu'}, {'units':128,'activation':'tanh'}, {'units':10,'activation':'sigmoid'}]

model = Model(layers, x_train, ytrain)

model.fit(lr = 7, epochs = 20)

pred = model.predict(x_test[10])

print('Predicted Value : ', np.argmax(pred))
print('Expected Value : ', y_test[10])

plt.imshow(x_test[10].reshape(28,28))
plt.show()
