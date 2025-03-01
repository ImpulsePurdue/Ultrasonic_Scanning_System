
"""
Ultrasonic Scanner Control System
=================================
This script integrates the initialization and data acquisition procedures for the custom ultrasonic scanning system. 

Developed for research in ultrasonic evaluation of additively manufactured components.

Author: Harshith Kumar Adepu
Date: 2025-03-01
Affiliation: Purdue University, IMPULSE Research Group

Usage:
- Ensure the required hardware components (stepper motors, transducer, and data acquisition system) are connected properly.
- Adjust parameters such as scan range and step size before running the script.
- Execute the script to perform the scanning operation and save acquired data.

Modules:
- Motion Control: Configures and moves the scanner to desired positions.
- Data Acquisition: Collects ultrasonic signals at each position.
- Storage & Processing: Saves acquired data for analysis.

Developed for research in ultrasonic evaluation of additively manufactured components.
"""

# Import necessary libraries
import time
import numpy as np
import serial  # Assuming serial communication is used
import os
import pyvisa as visa


def gcode_homimg(ser):
    ser.write(str.encode("G90\r\n"))
    time.sleep(2) 
    ser.write(str.encode("G0 X1 Y1\r\n"))
    time.sleep(2) 
    ser.write(str.encode("G90\r\n"))
    time.sleep(2) 
    ser.write(str.encode("G28\r\n"))
    time.sleep(2)  

def gcode_move_to(ser, x, y):
    ser.write(str.encode("G90\r\n"))
    ser.write(f"G0 X{x} Y{y}\r\n".encode())
    time.sleep(2)

def fetch_and_save_waveform(directory, filename):
    try:
        if not os.path.exists(directory):                                               # Ensure the directory exists
            os.makedirs(directory)

        full_file_path = os.path.join(directory, filename)

        rm = visa.ResourceManager()
        oscilloscope = rm.open_resource('USB0::0x2A8D::0x1766::MY63420861::0::INSTR')   # Path to oscilloscope
        oscilloscope.timeout = 40000

        oscilloscope.write(':WAVeform:SOURce CHANnel1')                                 # Adjust channel as needed
        oscilloscope.write(':WAVeform:FORMat ASCii')                                    # Set data format, ASCII for simplicity

        oscilloscope.write(':WAVeform:XINCrement?')
        x_increment = float(oscilloscope.read())
        oscilloscope.write(':WAVeform:DATA?')
        raw_data = oscilloscope.read_raw()
    
        oscilloscope.close()                                                            # Close connection
        rm.close()

        # Process raw data to remove the header
        header_len = 2 + int(raw_data[1])                                               # '2 +' accounts for '#N', where N is the number of digits specifying the length
        data = raw_data[header_len:]                                                    # Skip header
        data_string = data.decode('ascii')                                              # Convert bytes to ASCII string if necessary
        voltage_values = np.fromstring(data_string, sep=',')                            # Convert string to numpy array

        # Generate the time values based on the x_increment
        time_values = np.arange(0, len(voltage_values) * x_increment, x_increment)

        # Adjust if the lengths do not match
        min_length = min(len(time_values), len(voltage_values))
        time_values = time_values[:min_length]
        voltage_values = voltage_values[:min_length]

        # Prepare the full dataset
        full_data = np.column_stack((time_values, voltage_values))

        # Save the data to a file in CSV format with headers
        np.savetxt(full_file_path, full_data, delimiter=',', header='second,Volt', comments='', fmt=['%e', '%e'])

    except visa.VisaIOError as e:
        raise Exception(f"Failed to communicate with the oscilloscope: {e}")
    except Exception as e:
        print(f"Error: {e}")
        return None



x = 20
y = 30

def main():
    try:
        with serial.Serial('COM4', 115200, timeout=1) as ser:
            time.sleep(2)                                                               # Wait for the connection to initialize
            gcode_homimg(ser)
            print("Completed homing.")

    except serial.SerialException as e:
        print(f"Error opening or using the serial port: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    ser.close()
if __name__ == "__main__":
    main()

x1 = 36
y1 = 22
x2 = 12
y2 = 46

step = 0.25    # Resolution mm

fetch_and_save_waveform('test/P2', f'P2_{int(x1*100)}_{int(y1*100)}.csv')

def main():
    try:
        with serial.Serial('COM4', 115200, timeout=1) as ser:
            time.sleep(2)
            for y in np.arange(y1, y2 + step, step):
                for x in np.arange(x1, x2 - step, -step):
                    gcode_move_to(ser, x, y)
                    fetch_and_save_waveform('test/P2', f'P2_{int(x*100)}_{int(y*100)}.csv') # Multiply by 100 to save the file as 3625 instead of 36.25
            print("Completed all movements.")

    except serial.SerialException as e:
        print(f"Error opening or using the serial port: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    ser.close()
if __name__ == "__main__":
    main()
