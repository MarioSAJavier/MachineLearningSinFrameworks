import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

class LogisticRegression():

    def __init__(self, learningRate=0.001, nIterations=1000):   # se inicia los parametros de la clase
        self.learningRate = learningRate
        self.nIterations = nIterations
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):                             #Ajusta el modelo y por cada n iteracion cambiando el peso y el bias para la funcion sigmoide
        nSamples, nFeatures = X.shape
        self.weights = np.zeros(nFeatures)
        self.bias = 0

        for _ in range(self.nIterations):
            linearRegression = np.dot(X, self.weights) + self.bias
            predictions = sigmoid(linearRegression)
            dw = (1/nSamples) * np.dot(X.T, (predictions - y))
            db = (1/nSamples) * np.sum(predictions-y)

            self.weights = self.weights - self.learningRate*dw
            self.bias = self.bias - self.learningRate*db

            loss = - (1 / nSamples) * np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
            self.loss_history.append(loss)


    def predict(self, X):                              #Se hace la prediccion determinandolo por donde se encuentre el valor en la funcion sigmoide final obtenida del def init()
        linearRegression = np.dot(X, self.weights) + self.bias
        yPredicted = sigmoid(linearRegression)
        classPrediction = [0 if y<=0.5 else 1 for y in yPredicted]
        return classPrediction

    def get_loss_history(self):
        return self.loss_history