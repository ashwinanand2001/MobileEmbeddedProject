
#This file will read the signature data from the csv file and perform calculations to
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def find_start_end(roc):
    print(len(roc))
    for i in range(0,len(roc)):
        if roc[i] > .5:
            start = i
            break
    for i in reversed(range(0,len(roc))):
        if roc[i] > .5:
            end = i
            break
    return [start, end]

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
df = pd.read_csv("Skandan1.csv")
df.apply(pd.to_numeric)


d_total = average_roc(df)
[start, end] = find_start_end(d_total)
df.iloc[start:end].plot()
plt.show()