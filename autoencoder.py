import numpy as np
import os
import csv

from sklearn.preprocessing import MinMaxScaler
from keras import layers, losses, Sequential
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

scaler = MinMaxScaler()
filenames = []
for filename in sorted(os.listdir('dataset')):
    filenames.append(filename)

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

# uses autoencoder to predict if a file is an anomaly
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

# for each device
for i in range(9):
    # create a training set using the benign dataset
    benign = np.loadtxt(f'dataset/{i+1}.benign.csv', delimiter=",", skiprows=1)
    X_train = benign[:40000]
    # create and write data to a csv file specific to the device
    with open(f"statistics/autoencoder/device{i+1}.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(filenames)
        # run 10 times to get an average
        for ii in range(10):
            datapoints = []
            x = scaler.fit_transform(X_train)
            
            # create an autoencoder model
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
            
            with open(f"statistics/autoencoder/device{i}epoch.csv", "w") as g:
                writer = csv.writer(g)
                writer.writerow(monitor.stopped_epoch + 1)
            
            # save the autoencoder model to apply transfer learning later
            ae.save(f'neuralnetworks/device{i+1}/run{ii+1}')
            
            training_loss = losses.mse(x, ae(x))
            threshold = np.mean(training_loss)+np.std(training_loss)
            
            # test the model against every file in the dataset
            for filename in sorted(os.listdir('dataset')):
                file = np.loadtxt('dataset/' + filename, delimiter=",", skiprows=1)
                outcome = predict(file, threshold)
                print(filename)
                print_stats(file, outcome)
            writer.writerow(datapoints)
