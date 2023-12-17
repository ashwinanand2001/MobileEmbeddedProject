#!/usr/bin/env python3
import serial.tools.list_ports


# List available ports and select the one to use
ports = serial.tools.list_ports.comports()
port_list = [str(p) for p in ports]
print("Available ports:")
for p in port_list:
    print(p)
