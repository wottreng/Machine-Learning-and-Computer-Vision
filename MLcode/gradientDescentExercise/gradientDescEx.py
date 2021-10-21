"""
code credit goes to Mariya:
https://github.com/MariyaSha/
https://www.youtube.com/c/PythonSimplified
"""
import numpy as np


# Activation Function
def sigmoid(w_sum):
    return 1 / (1 + np.exp(-w_sum))


# Get Model Output
def get_prediction(features, weights, bias):
    return sigmoid(np.dot(features, weights) + bias)


# Loss Function
def cross_entropy(target, pred):
    return -(target * np.log10(pred) + (1 - target) * (np.log10(1 - pred)))


# Update Weights

def gradient_descent(feature, weights, target, prediction, l_rate, bias):
    # find new weights and bias
    new_weights = []
    bias += l_rate * (target - prediction)
    for x, w in zip(feature, weights):
        new_w = w + l_rate * (target - prediction) * x
        new_weights.append(new_w)
    # return updated values
    return new_weights, bias


# Data
features = np.array(([0.1, 0.5, 0.2], # result = 0
                     [0.2, 0.3, 0.1], # result = 1
                     [0.7, 0.4, 0.2], # result = 0
                     [0.1, 0.4, 0.3])) # result = 1

targets = np.array([0, 1, 0, 1])
random_weights = True
if random_weights:
    rg = np.random.default_rng()
    weights = rg.random((1, 3))[0]
else:
    weights = np.array([0.4, 0.2, 0.6])
bias = 0.5
l_rate = 0.01
num_of_epoch = 20

# Gradient Descent Steps Over number of Epochs
for epoch in range(num_of_epoch):
    for x, y in zip(features, targets):
        pred = get_prediction(x, weights, bias)
        error = cross_entropy(y, pred)
        weights, bias = gradient_descent(x, weights, y, pred, l_rate, bias)

    # Calculate and Print Average Loss
    predictions = get_prediction(features, weights, bias)
    average_loss = np.mean(cross_entropy(targets, predictions))
    print("**********----*******************")
    print("EPOCH", str(epoch))
    print("Average Loss: ", average_loss)
    print("***********----******************")
