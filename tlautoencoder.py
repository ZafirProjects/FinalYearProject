import os
import csv
import numpy as np

from keras import models, losses
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

z = 0
y = 0
scaler = MinMaxScaler()
filenames = []
for filename in sorted(os.listdir('dataset')):
    filenames.append(filename)

def predict(x, threshold, window_size=82):
    x = scaler.transform(x)
    predictions = losses.mse(x, ae(x)) > threshold
    # Majority voting over `window_size` predictions
    return np.array([np.mean(predictions[i-window_size:i]) > 0.5
                     for i in range(window_size, len(predictions)+1)])

def print_stats(data, outcome, datapoints):
    print(f"Shape of data: {data.shape}")
    print(f"Detected anomalies: {np.mean(outcome)*100}%")
    print()
    datapoints.append(np.mean(outcome)*100)

def train(ae, stage):
    # create a training set using the benign dataset
    benign = np.loadtxt(f'dataset/{z+1}.benign.csv', delimiter=",", skiprows=1)
    X_train = benign[:40000]
    # create and write data to a csv file specific to the device
    with open(f"statistics/{stage}/{i}.csv", 'a') as f:
        writer = csv.writer(f)
        if os.stat(f"statistics/{stage}/{i}.csv").st_size == 0:
            writer.writerow(filenames)
        # run 10 times to get an average
        datapoints = []
        x = scaler.fit_transform(X_train)
        
        # create an autoencoder model
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
        
        with open(f"statistics/{stage}/deviceepoch{z+1}.csv", "a") as g:
            writer2 = csv.writer(g)
            writer2.writerow([monitor.stopped_epoch + 1])
        
        training_loss = losses.mse(x, ae(x))
        threshold = np.mean(training_loss)+np.std(training_loss)
        
        # test the model against every file in the dataset
        for filename in sorted(os.listdir('dataset')):
            continue
            file = np.loadtxt('dataset/' + filename, delimiter=",", skiprows=1)
            outcome = predict(file, threshold)
            print(filename)
            print_stats(file, outcome, datapoints)
        writer.writerow(datapoints)

def save_epochs(monitor, stage):
    with open(f"statistics/{stage}/epoch{i}.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow([monitor.stopped_epoch])

def tlencoder(ae):
    if y == 0:
        ae.get_layer('sequential').get_layer('dense').trainable = False
        ae.get_layer('sequential').get_layer('dense_1').trainable = False
        ae.get_layer('sequential').get_layer('dense_2').trainable = False
    else:
        ae.get_layer(f'sequential_{2*y}').get_layer(f'dense_{9*y}').trainable = False
        ae.get_layer(f'sequential_{2*y}').get_layer(f'dense_{(9*y)+1}').trainable = False
        ae.get_layer(f'sequential_{2*y}').get_layer(f'dense_{(9*y)+2}').trainable = False
    train(ae, "encodertl")

def tlbottleneck(ae):
    if y == 0:
        ae.get_layer('sequential').get_layer('dense_3').trainable = False
        ae.get_layer('sequential').get_layer('dense_4').trainable = False
    else:
        ae.get_layer(f'sequential_{2*y}').get_layer(f'dense_{(9*y)+3}').trainable = False
        ae.get_layer(f'sequential_{2*y}').get_layer(f'dense_{(9*y)+4}').trainable = False
        ae.get_layer(f'sequential_{(2*y)+1}').get_layer(f'dense_{(9*y)+5}').trainable = False
    train(ae, "bottlenecktl")
    
def tldecoder(ae):
    ae.get_layer(f'sequential_{(2*y)+1}').get_layer(f'dense_{(9*y)+6}').trainable = False
    ae.get_layer(f'sequential_{(2*y)+1}').get_layer(f'dense_{(9*y)+7}').trainable = False
    ae.get_layer(f'sequential_{(2*y)+1}').get_layer(f'dense_{(9*y)+8}').trainable = False
    train(ae, "decodertl")

for i in sorted(os.listdir('neuralnetworks')):
    for ii in range(10):
        ae = models.load_model(f"neuralnetworks/{i}/run{ii+1}")
        tlencoder(ae)
        ae = models.load_model(f"neuralnetworks/{i}/run{ii+1}")
        tlbottleneck(ae)
        ae = models.load_model(f"neuralnetworks/{i}/run{ii+1}")
        tldecoder(ae)
        y = y + 1
    z = z + 1
