import os
import numpy as np

from keras import models, losses
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
ae = models.load_model("neuralnetworks/device1/run1")

benign = np.loadtxt('dataset/1.benign.csv', delimiter=",", skiprows=1)
X_train = benign[:40000]

x = scaler.fit_transform(X_train)

training_loss = losses.mse(x, ae(x))
threshold = np.mean(training_loss)+np.std(training_loss)

def predict(x, threshold, window_size=82):
    x = scaler.transform(x)
    predictions = losses.mse(x, ae(x)) > threshold
    # Majority voting over `window_size` predictions
    return np.array([np.mean(predictions[i-window_size:i]) > 0.5
                     for i in range(window_size, len(predictions)+1)])

def print_stats(data, outcome):
    print(f"Shape of data: {data.shape}")
    print(f"Detected anomalies: {np.mean(outcome)*100}%")
    print()

for filename in sorted(os.listdir('dataset')):
    file = np.loadtxt('dataset/' + filename, delimiter=",", skiprows=1)
    outcome = predict(file, threshold)
    print(filename)
    print_stats(file, outcome)
