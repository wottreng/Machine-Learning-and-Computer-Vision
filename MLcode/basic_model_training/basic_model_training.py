"""
code credit goes to Mariya:
https://github.com/MariyaSha/
https://www.youtube.com/c/PythonSimplified
"""
import pandas as pd
from functions import generate_data, get_weighted_sum, sigmoid, cross_entropy, update_weights, update_bias

bias = 0.5
l_rate = 0.01
epochs = 30
epoch_loss = []
features = 40  # rows
dataColumnsInFeature = 7  # columns

data, weights = generate_data(features, dataColumnsInFeature)  # row, columns

def train_model(data, weights, bias, l_rate, epochs):
    for e in range(epochs):
        individual_loss = []    
        for i in range(len(data)):
            feature = data.loc[i][:-1]
            target = data.loc[i][-1]
            w_sum = get_weighted_sum(feature, weights, bias)
            prediction = sigmoid(w_sum)
            loss = cross_entropy(target, prediction)
            individual_loss.append(loss)
            # gradient descent
            weights = update_weights(weights, l_rate, target, prediction, feature)
            bias = update_bias(bias, l_rate, target, prediction)
        average_loss = sum(individual_loss)/len(individual_loss)
        epoch_loss.append(average_loss)
        print("****************************")
        print("epoch", e)
        print(average_loss)
        
train_model(data, weights, bias, l_rate, epochs)

# plot the average loss
df = pd.DataFrame(epoch_loss)
df_plot = df.plot(kind="line", grid=True).get_figure()
df_plot.savefig("Training_Loss.pdf")
