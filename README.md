# Neural-Network-from-Scratch
It's a simple Feedforward Neural Network implemented just using Numpy library.

Libraries :
Numpy : Matrix calculation and most of the code !!!.
Tensorflow: Dataset (MNIST).
Matplotlib : Sample Image Display.

Files :
archi.py : Contains the basic architecture of Layer(Dense) and Model.
test.py : Contains the sample test used on MNIST dataset.

Note :
You may change the architecture of the model other than that used in test.py.
Also can use a different datatset but remember dimensions.
x_train : (no_examples, features)
y_train : (no_examples, target)

Architecture of the model must be a list of dictionary containing values of no of units in the layer and activation function.
test.py already has converted one-hot encoding for MNIST you may use your knowledge and change it according to your dataset.
