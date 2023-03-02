import numpy as np
import os
import csv

from sklearn.preprocessing import MinMaxScaler
from keras import layers, losses, Sequential
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

filenames = []
for filename in sorted(os.listdir('dataset')):
    filenames.append(filename)

benign = np.loadtxt('dataset/1.benign.csv', delimiter=",", skiprows=1)
X_train = benign[:40000]

class Autoencoder(Model):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Sequential([
            layers.Dense(115, activation="relu"),
            layers.Dense(86, activation="relu"),
            layers.Dense(57, activation="relu"),
            layers.Dense(37, activation="relu"),
            layers.Dense(28, activation="relu")
        ])
        self.decoder = Sequential([
            layers.Dense(37, activation="relu"),
            layers.Dense(57, activation="relu"),
            layers.Dense(86, activation="relu"),
            layers.Dense(115, activation="sigmoid")
        ])
    
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

scaler = MinMaxScaler()

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
    datapoints.append(np.mean(outcome)*100)

with open("statistics/ae.csv", 'w') as f:
    writer = csv.writer(f)
    writer.writerow(filenames)
    for i in range(10):
        datapoints = []
        x = scaler.fit_transform(X_train)

        ae = Autoencoder()
        ae.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
        monitor = EarlyStopping(
            monitor='val_loss',
            min_delta=1e-9,
            patience=5,
            verbose=1,
            mode='auto'
        )
        ae.fit(
            x=x,
            y=x,
            epochs=800,
            validation_split=0.3,
            shuffle=True,
            callbacks=[monitor]
        )

        training_loss = losses.mse(x, ae(x))
        threshold = np.mean(training_loss)+np.std(training_loss)

        for filename in sorted(os.listdir('dataset')):
            file = np.loadtxt('dataset/' + filename, delimiter=",", skiprows=1)
            outcome = predict(file, threshold)
            print(filename)
            print_stats(file, outcome)
        writer.writerow(datapoints)