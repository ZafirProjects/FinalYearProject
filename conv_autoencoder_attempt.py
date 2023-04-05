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
            layers.Conv2D(115, (3, 3), activation="relu", input_shape=(1, 32,115), padding='same'),
            layers.Conv2D(86, (3, 3), activation="relu", padding='same'),
            layers.Conv2D(57, (3, 3), activation="relu", padding='same'),
            layers.Conv2D(37, (3, 3), activation="relu", padding='same'),
            layers.Conv2D(28, (3, 3), activation="relu", padding='same'),
        ])
        self.decoder = Sequential([
            layers.Conv2D(37, (3, 3), activation="relu"),
            layers.UpSampling2D((2, 2)),
            layers.Conv2D(57, (3, 3), activation="relu"),
            layers.UpSampling2D((2, 2)),
            layers.Conv2D(86, (3, 3), activation="relu"),
            layers.UpSampling2D((2, 2)),
            layers.Conv2D(115, (3, 3), activation="sigmoid")
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
    with open(f"statistics/conv/autoencoder/device{i+1}.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(filenames)
        # run 10 times to get an average
        for ii in range(5):
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
            
            with open(f"statistics/conv/autoencoder/device{i+1}epoch.csv", "a") as g:
                writer2 = csv.writer(g)
                writer2.writerow([monitor.stopped_epoch])
            
            # save the autoencoder model to apply transfer learning later
            ae.save(f'neuralnetworks/conv/device{i+1}/run{ii+1}')
            
            training_loss = losses.mse(x, ae(x))
            threshold = np.mean(training_loss)+np.std(training_loss)
            
            # test the model against every file in the dataset
            for filename in sorted(os.listdir('dataset')):
                file = np.loadtxt('dataset/' + filename, delimiter=",", skiprows=1)
                outcome = predict(file, threshold)
                print(filename)
                print_stats(file, outcome)
            writer.writerow(datapoints)
