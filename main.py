import numpy as np
import pandas as pd

def initialise(input_layer_size, hidden_layer_size, output_layer_size):
    rng = np.random.default_rng()
    w1 = rng.standard_normal((hidden_layer_size, input_layer_size)) * np.sqrt(2.0 / input_layer_size)
    b1 = rng.random((hidden_layer_size, 1))
    w2 = rng.standard_normal((output_layer_size, hidden_layer_size)) * np.sqrt(2.0 / hidden_layer_size)
    b2 = rng.random((output_layer_size, 1))
    return w1, b1, w2, b2

def forward(X, w1, b1, w2, b2):
    z1 = w1.dot(X) + b1
    a1 = ReLU(z1)
    z2 = w2.dot(a1) + b2
    a2 = softmax(z2)
    return z1, a1, z2, a2

def backward(X, y, z1, a1, a2, w2):
    dz2 = (a2 - y) / y.shape[1]
    dw2 = dz2.dot(a1.T) 
    db2 = np.sum(dz2, axis=1, keepdims=True) 
    dz1 = w2.T.dot(dz2) * ReLU_back(z1)
    dw1 = dz1.dot(X.T) 
    db1 = np.sum(dz1, axis=1, keepdims=True) 
    return dw1, db1, dw2, db2

def update_parameters(w1, b1, w2, b2, dw1, db1, dw2, db2, eta):
    w1 -= eta * dw1 
    b1 -= eta * db1
    w2 -= eta * dw2
    b2 -= eta * db2
    return w1, b1, w2, b2

def ReLU(z):
    return np.maximum(0, z)

def softmax(z):
    z -= np.max(z, axis=0, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

def ReLU_back(z):
    return (z > 0)

def one_hot_encode(y, output_layer_size):
    encoded_labels = np.zeros((output_layer_size, len(y)))  
    for i in range(len(y)):
        encoded_labels[y[i]][i] = 1
    return encoded_labels

def cross_entropy(a2, y):
    epsilon = 1e-12
    return -np.sum(y * np.log(a2 + epsilon)) / y.shape[1]

def predict_classes(a2):
    return np.argmax(a2, axis=0)

def calculate_accuracy(predictions, train_labels):
    return np.sum(predictions == train_labels) / train_labels.size

def train(X, y, input_layer_size, hidden_layer_size, output_layer_size, learning_rate, epochs):
    one_hot_y = one_hot_encode(y, output_layer_size)
    w1, b1, w2, b2 = initialise(input_layer_size, hidden_layer_size, output_layer_size)
    z1, a1, z2, a2 = forward(X, w1, b1, w2, b2)
    for i in range(epochs):
        dw1, db1, dw2, db2 = backward(X, one_hot_y, z1, a1, a2, w2)
        w1, b1, w2, b2 = update_parameters(w1, b1, w2, b2, dw1, db1, dw2, db2, learning_rate)
        z1, a1, z2, a2 = forward(X, w1, b1, w2, b2)
        if i % 10 == 9:
            loss = cross_entropy(a2, one_hot_y)
            predictions = predict_classes(a2)
            accuracy = calculate_accuracy(predictions, y)
            print(f"Epoch: {i+1}, Loss: {loss:.3f}, Accuracy: {accuracy:.1%}")
    return w1, b1, w2, b2

if __name__ == '__main__':
    train_data = np.array(pd.read_csv('mnist/train.csv')).T
    train_features = train_data[1:] / 255.0
    train_labels = train_data[0]
    input_layer_size = train_features.shape[0]
    hidden_layer_size = 64
    output_layer_size = len(np.unique(train_labels))
    learning_rate = 0.1
    epochs = 100
    w1, b1, w2, b2 = train(train_features, train_labels, input_layer_size, hidden_layer_size, output_layer_size, learning_rate, epochs)