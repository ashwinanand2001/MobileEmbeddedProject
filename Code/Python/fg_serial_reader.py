#!/usr/bin/env python3

import serial
import csv
import re
from serial.tools import list_ports

# List available serial ports
port = list(list_ports.comports())
for p in port:
    print(p.device)

# Connect to Arduino over serial
# Replace '/dev/ttySx' with the correct port identified from the above list
ser = serial.Serial('com3', 9600)

# Create CSV file for data
with open('calibration_data.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['xAccel', 'yAccel', 'zAccel', 'xGyro', 'yGyro', 'zGyro', 'xMag', 'yMag', 'zMag'])

    # Read and write data
    while True:
        try:
            line = ser.readline().decode().strip()
            if line:  # Update this condition to match your data format
                values = line.split(',')  # Split based on comma for CSV data
                csv_writer.writerow(values)
        except KeyboardInterrupt:
            break

# Close the serial connection
ser.close()
