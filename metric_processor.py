import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

headers=['Device', 'TPR', 'FPR', 'TNR', 'FNR', 'Accuracy', 'Precision']

with open('metrics/confusion_matrices/deep_confusion_matrix.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerow(headers)
    for i in range(9):
        data = pd.read_csv(f'statistics/deep/autoencoder/device{i+1}.csv')
        columns = data.filter(like=str(i+1))
        column_averages = columns.mean(axis=0)
        column_modes = columns.mode().iloc[0]
        column_range = columns.max() - columns.min()
        column_deviation = columns.std()
        
        result = pd.concat([column_averages,column_modes,column_range,column_deviation], axis=1)
        result.columns = ['Average', 'Mode', 'Range', 'Deviation']
        result.to_csv(f'metrics/confidence/deep/device{i+1}.csv')
        
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        ii=0
        for column in column_averages:
            if ii == 0:
                if column <= 50:
                    tn += 1
                else:
                    fp += 1
                    
            else:
                if column > 50:
                    tp += 1
                else:
                    fn += 1
            ii=ii+1
        # Create confusion matrix
        conf_mat = np.array([[tp, fp], [fn, tn]])

        # Print confusion matrix
        new_rows = [f'device{i+1}', (tp/(tp+tn+fp+fn-1))*100, (fp/1)*100, (tn/1)*100, (fn/(tp+tn+fp+fn-1))*100, ((tp+tn)/(tp+tn+fp+fn))*100, (tp/(tp+fp))*100]
        writer.writerow(new_rows)

for i in range(3):
    with open(f'metrics/confusion_matrices/encoder_device_{i+1}_econfusion_matrix.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        for ii in range(3):
            data = pd.read_csv(f'statistics/deep/encodertl/device{i+1}vsdevice{ii+1}.csv')
            columns = data.filter(like=str(ii+1))
            column_averages = columns.mean(axis=0)
            column_modes = columns.mode().iloc[0]
            column_range = columns.max() - columns.min()
            column_deviation = columns.std()
            
            result = pd.concat([column_averages,column_modes,column_range,column_deviation], axis=1)
            result.columns = ['Average', 'Mode', 'Range', 'Deviation']
            result.to_csv(f'metrics/confidence/encoder/device{i+1}vsdevice{ii+1}.csv')
            
            tp = 0
            fp = 0
            fn = 0
            tn = 0
            iii=0
            for column in column_averages:
                if iii == 0:
                    if column <= 50:
                        tn += 1
                    else:
                        fp += 1
                else:
                    if column > 50:
                        tp += 1
                    else:
                        fn += 1
                iii=iii+1
            # Create confusion matrix
            conf_mat = np.array([[tp, fp], [fn, tn]])

            # Print confusion matrix
            new_rows = [f'device{ii+1}', (tp/(tp+tn+fp+fn-1))*100, (fp/1)*100, (tn/1)*100, (fn/(tp+tn+fp+fn-1))*100, ((tp+tn)/(tp+tn+fp+fn))*100, (tp/(tp+fp))*100]
            writer.writerow(new_rows)

for i in range(3):
    with open(f'metrics/confusion_matrices/bottleneck_device_{i+1}_econfusion_matrix.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        for ii in range(3):
            data = pd.read_csv(f'statistics/deep/bottlenecktl/device{i+1}vsdevice{ii+1}.csv')
            columns = data.filter(like=str(ii+1))
            column_averages = columns.mean(axis=0)
            column_modes = columns.mode().iloc[0]
            column_range = columns.max() - columns.min()
            column_deviation = columns.std()
            
            result = pd.concat([column_averages,column_modes,column_range,column_deviation], axis=1)
            result.columns = ['Average', 'Mode', 'Range', 'Deviation']
            result.to_csv(f'metrics/confidence/bottleneck/device{i+1}vsdevice{ii+1}.csv')
            
            tp = 0
            fp = 0
            fn = 0
            tn = 0
            iii=0
            for column in column_averages:
                if iii == 0:
                    if column <= 50:
                        tn += 1
                    else:
                        fp += 1
                else:
                    if column > 50:
                        tp += 1
                    else:
                        fn += 1
                iii=iii+1
            # Create confusion matrix
            conf_mat = np.array([[tp, fp], [fn, tn]])

            # Print confusion matrix
            new_rows = [f'device{ii+1}', (tp/(tp+tn+fp+fn-1))*100, (fp/1)*100, (tn/1)*100, (fn/(tp+tn+fp+fn-1))*100, ((tp+tn)/(tp+tn+fp+fn))*100, (tp/(tp+fp))*100]
            writer.writerow(new_rows)

for i in range(3):
    with open(f'metrics/confusion_matrices/decoder_device_{i+1}_econfusion_matrix.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        for ii in range(3):
            data = pd.read_csv(f'statistics/deep/decodertl/device{i+1}vsdevice{ii+1}.csv')
            columns = data.filter(like=str(ii+1))
            column_averages = columns.mean(axis=0)
            column_modes = columns.mode().iloc[0]
            column_range = columns.max() - columns.min()
            column_deviation = columns.std()
            
            result = pd.concat([column_averages,column_modes,column_range,column_deviation], axis=1)
            result.columns = ['Average', 'Mode', 'Range', 'Deviation']
            result.to_csv(f'metrics/confidence/decoder/device{i+1}vsdevice{ii+1}.csv')
            
            tp = 0
            fp = 0
            fn = 0
            tn = 0
            iii=0
            for column in column_averages:
                if iii == 0:
                    if column <= 50:
                        tn += 1
                    else:
                        fp += 1
                else:
                    if column > 50:
                        tp += 1
                    else:
                        fn += 1
                iii=iii+1
            # Create confusion matrix
            conf_mat = np.array([[tp, fp], [fn, tn]])

            # Print confusion matrix
            new_rows = [f'device{ii+1}', (tp/(tp+tn+fp+fn-1))*100, (fp/1)*100, (tn/1)*100, (fn/(tp+tn+fp+fn-1))*100, ((tp+tn)/(tp+tn+fp+fn))*100, (tp/(tp+fp))*100]
            writer.writerow(new_rows)
            
with open('metrics/epochs/autoencoder_epochs.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerow(['Device', 'Average'])
    for i in range(9):
        data = pd.read_csv(f'statistics/deep/autoencoder/device{i+1}epoch.csv')
        average = data.loc[0].mean()
        writer.writerow([f'Device{i+1}', average])

for i in range(3):
    with open(f'metrics/epochs/encdoder_device{i+1}.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['Device', 'Average'])
        data = np.loadtxt(f'statistics/deep/encodertl/device{i+1}epoch.csv', delimiter=',', usecols=[0])
        device1=np.array([])
        device2=np.array([])
        device3=np.array([])
        ii = 0
        for line in data:
            if ii % 3 == 0:
                device1 = np.append(device1, int(line))
            if ii % 3 == 1:
                device2 = np.append(device2, int(line))
            if ii % 3 == 2:
                device3 = np.append(device3, int(line))
            ii+=1
            
        writer.writerow([f'device1', np.mean(device1)])
        writer.writerow([f'device2', np.mean(device2)])
        writer.writerow([f'device3', np.mean(device3)])

for i in range(3):
    with open(f'metrics/epochs/bottleneck_device{i+1}.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['Device', 'Average'])
        data = np.loadtxt(f'statistics/deep/bottlenecktl/device{i+1}epoch.csv', delimiter=',', usecols=[0])
        device1=np.array([])
        device2=np.array([])
        device3=np.array([])
        ii = 0
        for line in data:
            if ii % 3 == 0:
                device1 = np.append(device1, int(line))
            if ii % 3 == 1:
                device2 = np.append(device2, int(line))
            if ii % 3 == 2:
                device3 = np.append(device3, int(line))
            ii+=1
        writer.writerow([f'device1', np.mean(device1)])
        writer.writerow([f'device2', np.mean(device2)])
        writer.writerow([f'device3', np.mean(device3)])

for i in range(3):
    with open(f'metrics/epochs/decdoder_device{i+1}.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['Device', 'Average'])
        data = np.loadtxt(f'statistics/deep/decodertl/device{i+1}epoch.csv', delimiter=',', usecols=[0])
        device1=np.array([])
        device2=np.array([])
        device3=np.array([])
        ii = 0
        for line in data:
            if ii % 3 == 0:
                device1 = np.append(device1, int(line))
            if ii % 3 == 1:
                device2 = np.append(device2, int(line))
            if ii % 3 == 2:
                device3 = np.append(device3, int(line))
            ii+=1
        writer.writerow([f'device1', np.mean(device1)])
        writer.writerow([f'device2', np.mean(device2)])
        writer.writerow([f'device3', np.mean(device3)])