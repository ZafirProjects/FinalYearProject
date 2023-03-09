import os
import numpy as np

from keras import models, losses
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
ae = models.load_model("neuralnetworks/device1/run1")

layer1 = ae.get_layer('sequential').get_layer('dense')
layer2 = ae.get_layer('sequential').get_layer('dense_1')
layer3 = ae.get_layer('sequential').get_layer('dense_2')
layer4 = ae.get_layer('sequential').get_layer('dense_3')
layer5 = ae.get_layer('sequential').get_layer('dense_4')
layer6 = ae.get_layer('sequential_1').get_layer('dense_5')
layer7 = ae.get_layer('sequential_1').get_layer('dense_6')
layer8 = ae.get_layer('sequential_1').get_layer('dense_7')
layer9 = ae.get_layer('sequential_1').get_layer('dense_8')

layer4.trainable = False
layer5.trainable = False
layer6.trainable = False

benign = np.loadtxt('dataset/1.benign.csv', delimiter=",", skiprows=1)
X_train = benign[:40000]

x = scaler.fit_transform(X_train)

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

print(monitor.stopped_epoch + 1)

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
