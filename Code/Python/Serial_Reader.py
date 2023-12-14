#!/usr/bin/env python3

import serial
import csv
import pandas as pd
import re
from serial.tools import list_ports
import tkinter
# m=tkinter.Tk()
# stop = tkinter.Button(m, text='Stop', width=25, height=25, bg= 'red')
# run = tkinter.Button(m, text='Run', width=25, height=25, bg= 'red')
# stop.pack()
# run.pack()
# m.mainloop()
port = list(list_ports.comports())
for p in port:
    print(p.device)

# Connect to Arduino over serial
ser = serial.Serial('/dev/cu.usbmodem1101', 9600)  # Change 'COM3' to the appropriate port

# Create CSV file for data
with open('Skandan10.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['xAccel', 'yAccel', 'zAccel', 'xGyro', 'yGyro', 'zGyro', 'xMag', 'yMag', 'zMag'])  # Add more headers if needed

    # Read and write data
    while True:
        try:
            line = ser.readline().decode().strip()
            #print(line)
            if re.match('-[0-9].', line):
                values = line.split(" ")
                #print(values)
                csv_writer.writerow(values)
        except KeyboardInterrupt:
            break

# Close the serial connection
ser.close()
