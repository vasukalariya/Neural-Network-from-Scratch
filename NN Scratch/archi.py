import numpy as np


class Dense():

    def __init__(self, no_examples, no_units, prev, activation):
        
        self.no_examples = no_examples
        self.no_units = no_units
        self.neurons = np.random.randn(no_examples, self.no_units)
        self.weights = np.random.randn(prev, self.no_units)*0.01
        self.bias = np.zeros((1, self.no_units))*0.01
        self.activation = activation
        self.dweights = np.zeros(self.weights.shape)
        self.dbias = np.zeros(self.bias.shape)
        self.error = None


    def activate(self):

        if self.activation == 'sigmoid':
            self.neurons = 1.0 / (1.0 + np.exp(-self.neurons))

        elif self.activation == 'tanh':
            self.neurons = np.tanh(self.neurons)

        elif self.activation == 'relu':
            self.neurons = np.maximum(0, self.neurons)

        else:
            pass

    def deractivation(self):
        
        if self.activation == 'sigmoid':
            self.neurons = self.neurons*(1 - self.neurons)

        elif self.activation == 'tanh':
            self.neurons = 1 - np.square(self.neurons)
        
        elif self.activation == 'relu':
            self.neurons[self.neurons <= 0] = 0
            self.neurons[self.neurons > 0] = 1
        
        else:
            pass

class Model():

    def __init__(self, layers, xdata, ydata):
        model = []
        self.cost = None
        self.xdata = xdata
        self.ydata = ydata

        for i,layer in enumerate(layers):
            
            if i != 0:
                l = Dense(model[-1].no_examples, layer['units'], model[-1].no_units, layer['activation'])
            else:
                l = Dense(self.xdata.shape[0], layer['units'], self.xdata.shape[1], layer['activation'])

            model.append(l)
        
        self.model = model

    def forward(self, x_data):

        for i,layer in enumerate(self.model):
               
            if i == 0:
                layer.neurons = np.dot(x_data, layer.weights) + layer.bias
            else:
                layer.neurons = np.dot(self.model[i-1].neurons, layer.weights) + layer.bias
                
            layer.activate()

        cost = np.sum(np.square(self.ydata - self.model[-1].neurons))
        #cost = -(self.ydata*np.log(self.model[-1].neurons) + (1 - self.ydata)*np.log(1 - self.model[-1].neurons))
        #cost = np.sum(cost)

        cost = cost / self.model[-1].no_examples
        self.cost = cost

        return self.model[-1].neurons, cost

    def backward(self):

        error = (self.ydata - self.model[-1].neurons)
        #error = -self.ydata/self.model[-1].neurons + (1 - self.ydata)/(1 - self.model[-1].neurons)

        mlen = len(self.model)-1

        for i,layer in reversed(list(enumerate(self.model))):

            if i == mlen:

                layer.deractivation()
                error = layer.neurons*error
                layer.dweights = np.dot(np.transpose(self.model[i-1].neurons), error)
                
            else:
                
                layer.deractivation()
                error = np.dot(error, np.transpose(self.model[i+1].weights))
                error = layer.neurons*error
                
                if i != 0:
                    layer.dweights = np.dot(np.transpose(self.model[i-1].neurons), error)
                
                else:
                    layer.dweights = np.dot(np.transpose(self.xdata), error)

            layer.dbias = np.sum(error, axis = 0)

    def fit(self, lr, epochs):

        print('Model Training ...')

        for i in range(epochs):
            
            print('\nEpoch : ',i)
            _, Cost = self.forward(self.xdata)
            print('Cost : ',Cost)

            self.backward()

            for i,layer in enumerate(self.model):

                layer.weights = layer.weights + lr*layer.dweights/self.xdata.shape[0]
                layer.bias = layer.bias + lr*layer.dbias/self.xdata.shape[0]

        print('\nModel Trained !!!\n')

    def predict(self, xtdata):
        pred,abc = self.forward(xtdata)

        return pred
