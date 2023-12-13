#!/usr/bin/env python3

import serial
import csv
import pandas as pd
import re
from serial.tools import list_ports

port = list(list_ports.comports())
for p in port:
    print(p.device)

# Connect to Arduino over serial
ser = serial.Serial('/dev/cu.usbmodem1101', 9600)  # Change 'COM3' to the appropriate port

# Create CSV file for data
with open('data.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['xAccel, yAccel, zAccel, xGyro, yGyro, zGyro, xMag, yMag, zMag'])  # Add more headers if needed

    # Read and write data
    while True:
        try:
            line = ser.readline().decode().strip()
            if re.match('-[0-9].', line):
                # Print received line
                #print(line)

                # Write to CSV
                csv_writer.writerow([line])
        except KeyboardInterrupt:
            break

# Close the serial connection
ser.close()

# Analyze data using pandas
df = pd.read_csv('data.csv')
print(df)
# Perform analysis on df (e.g., plot, statistics, etc.)
