#This file will read the signature data from the csv file and perform calculations to
from re import X
from tkinter import Y
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pywt
from scipy.spatial.distance import euclidean
from tensorflow import keras
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
import glob
import os
import csv
import math

def read_sensor_data(signer = ''):
    #reading the files in directory
    inp_data = os.listdir(path="/Users/skan/Documents/VSCode/MobileEmbeddedProject/SignatureData")

    for val in inp_data:
        if ".csv" not in val:
            inp_data.remove(val)
        if signer != '' and signer not in val:
            inp_data.remove(val)

    #extracting data from the files
    sensor_data = []
    signers = []

    for file in inp_data:
        df = pd.read_csv(os.path.join("/Users/skan/Documents/VSCode/MobileEmbeddedProject/SignatureData/", file))
        df.apply(pd.to_numeric)
        df = trim_sensor_data(df)
        sensor_data.append(df)
        signers.append(file.split(".")[0].rstrip('0123456789'))

    create_sig_meta_file(sensor_data, signers)
    return [sensor_data, signers]

def create_sig_meta_file(sensor_data, signers):
    #signers_flat = pd.Series(signers).drop_duplicates().tolist()

    temp_dict = {}
    for i in range(len(signers)):
        #print(signers[i])
        if signers[i] not in temp_dict:
            temp_dict[signers[i]] = [sensor_data[i]]
        else:
            temp_dict[signers[i]].append(sensor_data[i])


    with open("signer_meta_data.csv", "w", newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Name', 'Mean Signature Length', 'Avg STD Deviation Length'])
        for signer in temp_dict:
            [meanSigLength, stdDeviation] = standard_deviation_signature_length(temp_dict[signer], len(filter(bool, temp_dict[signer])))
            csv_writer.writerow([signer, meanSigLength, stdDeviation])
        

def standard_deviation_signature_length(x_values, numValues):
    meanSigLength = 0
    stdDeviation = 0
    if numValues > 1:
        for value in x_values:
            meanSigLength += value.shape[0]
        meanSigLength /= len(x_values)
        sumSquares = 0
        for value in x_values:
            sumSquares += pow(value.shape[0] - meanSigLength, 2)
        stdDeviation = sumSquares/(len(x_values)-1)
        stdDeviation = math.sqrt(stdDeviation)
    else:
        meanSigLength += x_values.shape[0]
        stdDeviation = 0

    return [meanSigLength, stdDeviation]

def predictSignature(file):
    
    inp_data = os.path.join("/Users/skan/Documents/VSCode/MobileEmbeddedProject/SignatureData/",file)

    # if ".csv" not in inp_data:
    #     inp_data.remove(val)

    #extracting data from the files
    df = pd.read_csv(inp_data)
    df.apply(pd.to_numeric)
    df = trim_sensor_data(df)
    signer = (file.split(".")[0].rstrip('0123456789'))
    [meanSigLength, stdDeviation] = standard_deviation_signature_length(df, 1)

    meta_data = pd.read_csv("signer_meta_data.csv")
    meta_data.set_index('Name', inplace=True)
    #print(meta_data)
    #print(signer)
    if meanSigLength > meta_data.loc[signer]["Mean Signature Length"] + 2*meta_data.loc[signer]["Avg STD Deviation Length"] \
        or meanSigLength < meta_data.loc[signer]["Mean Signature Length"] - 2*meta_data.loc[signer]["Avg STD Deviation Length"]:
        return "User Authentication Failed"
    

    return "User Authentication Success"

#def evaluateModel():


def train_model(x_train, y_train):
    sequence_length = x_train[0].shape[0]  # Adjust based on your data
    num_features = x_train[0].shape[1]
    x_train_reshaped = x_train.reshape((x_train.shape[0], sequence_length, num_features))
    num_classes = 1

    model = Sequential()
    model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=(sequence_length, num_features)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train_reshaped, y_train, epochs=num_epochs, batch_size=batch_size, validation_split=0.2, verbose=2)

    return model

def smoothData(df):
    window_size = 10
    df_smoothed = df.rolling(window=window_size).mean().dropna()
    return df_smoothed

def extractWavelets(df):
    wavelet = 'db1'
    level = 4
    # Perform wavelet decomposition
    x_accel_coeffs = pywt.downcoef(df['xAccel'], wavelet, level=level)
    y_accel_coeffs = pywt.downcoef(df['yAccel'], wavelet, level=level)
    z_accel_coeffs = pywt.downcoef(df['zAccel'], wavelet, level=level)
    x_gyro_coeffs = pywt.downcoef(df['xGyro'], wavelet, level=level)
    y_gyro_coeffs = pywt.downcoef(df['yGyro'], wavelet, level=level)
    z_gyro_coeffs = pywt.downcoef(df['zGyro'], wavelet, level=level)
    x_mag_coeffs = pywt.downcoef(df['xMag'], wavelet, level=level)
    y_mag_coeffs = pywt.downcoef(df['yMag'], wavelet, level=level)
    z_mag_coeffs = pywt.downcoef(df['zMag'], wavelet, level=level)

    return [x_accel_coeffs, y_accel_coeffs, z_accel_coeffs, x_gyro_coeffs, y_gyro_coeffs, z_gyro_coeffs, x_mag_coeffs, y_mag_coeffs, z_mag_coeffs]


def find_start_end(roc):
    #print(len(roc))
    for i in range(0,len(roc)):
        if roc[i] > .5:
            start = i
            break
    for i in reversed(range(0,len(roc))):
        if roc[i] > .5:
            end = i
            break
    return [start, end]

def trim_sensor_data(df):
    d_total = average_roc(df)
    [start,end] = find_start_end(d_total)
    df = df.iloc[start:end]
    print(end-start)
    df = smoothData(df)
    return df

def average_roc(df):
    d_xAccel = np.gradient(df['xAccel'],df.index)
    d_yAccel = np.gradient(df['yAccel'],df.index)
    d_zAccel = np.gradient(df['zAccel'],df.index)
    d_xGyro = np.gradient(df['xGyro'],df.index)
    d_yGyro = np.gradient(df['yGyro'],df.index)
    d_zGyro = np.gradient(df['zGyro'],df.index)
    d_xMag = np.gradient(df['xMag'],df.index)
    d_yMag = np.gradient(df['yMag'],df.index)
    d_zMag = np.gradient(df['zMag'],df.index)
    d_total = np.zeros(len(df.index))
    for i in df.index:
        d_total[i] = (d_xAccel[i]+d_yAccel[i]+d_zAccel[i]+d_xGyro[i]+d_yGyro[i]+d_zGyro[i]+d_xMag[i]+d_yMag[i]+d_zMag[i])/9
    # plt.plot(df.index, d_total)
    # plt.show()
    return d_total

#[sensor_data, signers] = read_sensor_data()

print(predictSignature("Skandan11.csv"))

#split input data into training and test data - 50% split
# splitPercent = .5
# index = splitPercent*len(sensor_data) #same for x_traina and y_train
# x_train = sensor_data[0:index]
# y_train = signers[0:index]
# x_test = sensor_data[index+1:len(sensor_data)-1]
# y_test = signers[index+1:len(sensor_data)-1]

#train model
#train_model()


# df = pd.read_csv("Skandan1.csv")
# df.apply(pd.to_numeric)
# d_total = average_roc(df)
# [start, end] = find_start_end(d_total)
# df = smoothData(df)
# df.iloc[start:end].plot()
# df = df.iloc[start:end]
# plt.show()

#[df_x_accel_coeffs, df_accel_coeffs, df_z_accel_coeffs, df_x_gyro_coeffs, df_y_gyro_coeffs, df_z_gyro_coeffs, df_x_mag_coeffs, df_y_mag_coeffs, df_z_mag_coeffs] = extractWavelets(df)
#print(df_x_accel_coeffs)
#plt.plot(df_x_accel_coeffs)
# df1 = pd.read_csv("Skandan2.csv")
# df1.apply(pd.to_numeric)
# d_total = average_roc(df1)
# [start, end] = find_start_end(d_total)
# df1 = df1.iloc[start:end]

#[df1_x_accel_coeffs, df1_accel_coeffs, df1_z_accel_coeffs, df1_x_gyro_coeffs, df1_y_gyro_coeffs, df1_z_gyro_coeffs, df1_x_mag_coeffs, df1_y_mag_coeffs, df1_z_mag_coeffs] = extractWavelets(df1)

#distance = euclidean(df_x_accel_coeffs, df1_x_accel_coeffs)