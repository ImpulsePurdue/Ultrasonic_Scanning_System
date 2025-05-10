#!/usr/bin/env python
# coding: utf-8

# C-SCAN MAPPING
# 
# 
# Written by: HARSHITH KUMAR ADEPU (05/06/2024)

# In[2]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
from matplotlib import scale as mscale
from mpl_toolkits.mplot3d import axes3d
from matplotlib.colors import LightSource
from scipy.interpolate import griddata
from scipy.optimize import fsolve, curve_fit
from sympy import symbols, Matrix
from scipy.special import jv, yv, hankel2
from scipy.integrate import simps
from matplotlib.patches import Rectangle
from scipy.signal import butter, filtfilt, medfilt, savgol_filter, find_peaks
from scipy.signal import find_peaks, hilbert, correlate, correlation_lags, resample, savgol_filter, tukey, hann
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# In[3]:


def calculate_correlation(pulse_1, pulse_2):
    """
    Calculates the cross-correlation between two pulses.

    Parameters:
    pulse_1 (array): The first pulse array.
    pulse_2 (array): The second pulse array.

    Returns:
    tuple: A tuple containing:
        - correlation (array): The correlation values of the two pulses.
        - lags (array): The corresponding lags for the correlation values.
        - max_corr_index (int): The index of the maximum correlation value.
    """
    correlation = correlate(pulse_2, pulse_1, mode='full')
    lags = np.arange(-len(pulse_1) + 1, len(pulse_2))
    max_corr_index = np.argmax(correlation)
    return correlation, lags, max_corr_index

def quadratic_fit(lags, correlation, max_corr_index, fit_range=2):
    """
    Performs a quadratic fit around the maximum correlation point.

    Parameters:
    lags (array): The lags corresponding to the correlation values.
    correlation (array): The correlation values.
    max_corr_index (int): The index of the maximum correlation value.
    fit_range (int, optional): The number of points around the peak to use for fitting (default is 2).

    Returns:
    float: The vertex of the quadratic fit, representing the refined peak location.
    """
    x_fit = lags[max_corr_index - fit_range: max_corr_index + fit_range + 1]
    y_fit = correlation[max_corr_index - fit_range: max_corr_index + fit_range + 1]
    coefficients = np.polyfit(x_fit, y_fit, 2)
    a, b, _ = coefficients
    vertex = -b / (2 * a)
    return vertex

def xcorr_cpi(pulse_1, pulse_2, f_sampling):
    """
    Computes the cross-correlation time delay with sub-sample accuracy using a quadratic fit.

    Parameters:
    pulse_1 (array): The first pulse array.
    pulse_2 (array): The second pulse array.
    f_sampling (float): The sampling frequency.

    Returns:
    float: The time delay in seconds with sub-sample accuracy, or NaN if one of the pulses is all zeros.

    Raises:
    ValueError: If there are not enough data points around the peak for a reliable fit.
    """
    if np.all(pulse_1 == 0) or np.all(pulse_2 == 0):
        return np.nan

    correlation, lags, max_corr_index = calculate_correlation(pulse_1, pulse_2)
    if max_corr_index < 2 or max_corr_index + 2 >= len(correlation):
        raise ValueError("Not enough data points around the peak for a reliable fit.")
    vertex_cpi = quadratic_fit(lags, correlation, max_corr_index)
    time_delay_seconds_cpi = vertex_cpi / f_sampling
    return time_delay_seconds_cpi


# Input the following details from your experimental setup

# In[4]:


sample_thickness = 21.81 * 1e-3 
x1, y1 = 36, 22
x2, y2 = 14, 44
step = 0.5


# In[5]:


directory = '01_Scan_Data'  # Define the directory containing the CSV files

data = {}

x_values = set()
y_values = set()

for filename in os.listdir(directory):
    if filename.startswith("FE1_11_") and filename.endswith(".csv"):
        parts = filename.split('_')
        x = int(parts[2])
        y = int(parts[3].replace('.csv', ''))
        x_values.add(x)
        y_values.add(y)

x_values = sorted(list(x_values))
y_values = sorted(list(y_values))

x_index = {val: idx for idx, val in enumerate(x_values)}
y_index = {val: idx for idx, val in enumerate(y_values)}

# Read one file to determine the length of the voltage data
sample_file = next(f for f in os.listdir(directory) if f.startswith("FE1_11_") and f.endswith(".csv"))
# Load ultrasonic waveform data from CSV file
sample_df = pd.read_csv(os.path.join(directory, sample_file), skiprows=2, usecols=[1], names=['Voltage'])
time_series_length = len(sample_df)

# Initialize a 3D NumPy array for voltage data
voltage_data = np.zeros((len(x_values), len(y_values), time_series_length))

# Populate the 3D array
for filename in os.listdir(directory):
    if filename.startswith("FE1_11_") and filename.endswith(".csv"):
        parts = filename.split('_')
        x = int(parts[2])
        y = int(parts[3].replace('.csv', ''))
        
        file_path = os.path.join(directory, filename)
# Load ultrasonic waveform data from CSV file
        df = pd.read_csv(file_path, skiprows=2, usecols=[1], names=['Voltage'])
        
        # Store voltage values in the array
        voltage_data[x_index[x], y_index[y], :] = df['Voltage'].to_numpy()

for i in range(voltage_data.shape[0]):  # Iterate over x
    for j in range(voltage_data.shape[1]):  # Iterate over y
        waveform = voltage_data[i, j, :]
        normalized = 2 * (waveform - np.min(waveform)) / (np.max(waveform) - np.min(waveform)) - 1
        centered = normalized - np.mean(normalized)
        voltage_data[i, j, :] = centered / max(abs(np.min(centered)), abs(np.max(centered)))


# In[6]:


# Load ultrasonic waveform data from CSV file
time_array = pd.read_csv(os.path.join(directory, sample_file), skiprows=2, usecols=[0], names=['Time'])['Time'].to_numpy()
f_sampling = 1 / (time_array[1] - time_array[0])
sample_interval             = 1 / f_sampling * 1e6  

c_sample                    = 5800                               # Estimated Wave speed in Sample (m/s)
lamda                       = c_sample / (7.5 * 1e6)                   # Convert f_td from MHz to Hz for calculation (m/wavelength)
pulse_width                 = (lamda * 3.75) * 2 / c_sample             # Approximate width of the pulse (seconds)
pulse_width_us              = pulse_width * 1e6                         # Approximate width of the pulse (micro seconds)
pulse_width_samples         = int(pulse_width_us * f_sampling * 1e-6)          # Approximate width of the pulse (micro seconds)

bw_distance_us              = (2 * sample_thickness / c_sample) * 1e6   # Expected separation between FW and BW pulses in microseconds
bw_distance_samples         = int(bw_distance_us * f_sampling * 1e-6)          # Convert µs to seconds and calculate samples


# In[7]:


isolated_signals_all        = np.zeros((voltage_data.shape[0], voltage_data.shape[1], 3, voltage_data.shape[2]))
peaks_all                   = np.full((voltage_data.shape[0], voltage_data.shape[1], 2), -1, dtype=int)  # Store 2 peaks for each (i, j)

for i in range(voltage_data.shape[0]):
    for j in range(voltage_data.shape[1]):
        
        t_min, t_max = 5 * 1e-6, 22 * 1e-6  # µs
        start_idx = np.searchsorted(time_array, t_min)
        end_idx = np.searchsorted(time_array, t_max)
        trimmed_wave = np.copy(voltage_data[i, j, :])
        trimmed_wave[:start_idx] = 0  # Set values before 5 µs to zero
        trimmed_wave[end_idx:] = 0    # Set values after 22 µs to zero

        smooth_wave = savgol_filter(trimmed_wave, window_length=15, polyorder=2)
        amplitude_envelope = np.abs(hilbert(smooth_wave))
        amplitude_envelope = savgol_filter(amplitude_envelope, window_length=75, polyorder=2)
        isolated_signals_all[i, j, 2, :] = amplitude_envelope  

        peaks, _ = find_peaks(amplitude_envelope, distance = bw_distance_samples, prominence=0.025)

        if len(peaks) >= 2:
            prominent_peaks = peaks[:2]  
            peaks_all[i, j, :] = prominent_peaks  
            
            for k, peak in enumerate(prominent_peaks):
                start = max(peak - pulse_width_samples, 0)
                end = min(peak + pulse_width_samples, voltage_data.shape[2])

                isolated_signals_all[i, j, k, start:end] = voltage_data[i, j, start:end]

        else:
            isolated_signals_all[i, j, 0, :] = 0  
            isolated_signals_all[i, j, 1, :] = 0  
            peaks_all[i, j, :] = -1  


# In[11]:


amplitude_results               = np.zeros((voltage_data.shape[0], voltage_data.shape[1]))
xcorr_pi_results                = np.zeros((voltage_data.shape[0], voltage_data.shape[1]))

for i in range(voltage_data.shape[0]):
    for j in range(voltage_data.shape[1]):
        fw_signal = isolated_signals_all[i, j, 0, :]
        bw_signal = isolated_signals_all[i, j, 1, :]

        try:
            time_delay_xcorr_pi     = xcorr_cpi(fw_signal, bw_signal, f_sampling)
            amplitude_results[i, j] = np.max(fw_signal)
            xcorr_pi_results[i, j]  = 2 * sample_thickness / time_delay_xcorr_pi
        except Exception as e:
            print(f"xcorr_pi failed at point ({i}, {j}): {e}")


# In[15]:


root = tk.Tk()
root.title("Wave Speed Analysis")

fig, axs = plt.subplots(2, 2, figsize=(15, 12))

im1 = axs[1, 0].imshow(amplitude_results, aspect='auto', cmap='jet')
axs[1, 0].set_title("Amplitude")
axs[1, 0].set_xlabel('Scan Points')
axs[1, 0].set_ylabel('Index Points')
cbar = fig.colorbar(im1, ax=axs[1, 0], fraction=0.046, pad=0.04)  # Using im1 as the reference for color scale
cbar.set_label('Amplitude')

im2 = axs[1, 1].imshow(xcorr_pi_results, aspect='auto', cmap='viridis')
axs[1, 1].set_title("Cross-Correlation with Parabolic Interpolation (CPI)")
axs[1, 1].set_xlabel('Scan Points')
axs[1, 1].set_ylabel('Index Points')

cbar = fig.colorbar(im2, ax=axs[1, 1], fraction=0.046, pad=0.04)  # Using im1 as the reference for color scale
cbar.set_label('Wavespeed')

# Plot initial waveform and signals (empty to start)
waveform_plot, = axs[0, 0].plot([], [], label="Waveform", color = 'C0')

axs[0, 0].legend()
signal1_plot, = axs[0, 1].plot([], [], label="Signal 1")
signal2_plot, = axs[0, 1].plot([], [], label="Signal 2")
axs[0, 1].legend()
axs[0, 1].set_ylim(-1,1)
axs[0, 0].set_ylim(-1,1)

# Set titles for waveform and signals
axs[0, 0].set_title("Original Waveform")
axs[0, 0].set_xlabel("Time ($\mu$s)")
axs[0, 0].set_ylabel("Amplitude")
axs[0, 1].set_title("Extracted Signals")
axs[0, 1].set_xlabel("Time ($\mu$s)")
axs[0, 1].set_ylabel("Amplitude")

# Function to update waveform and signal plots based on clicked pixel
def on_click(event):
    if event.inaxes == axs[1, 0]:  # Check if click is in the cross-correlation plot
        x_click = int(event.xdata)  # X coordinate (scan points)
        y_click = int(event.ydata)  # Y coordinate (index points)

        if 0 <= x_click < xcorr_pi_results.shape[1] and 0 <= y_click < xcorr_pi_results.shape[0]:
            # axs[0, 0].clear()  # Clear the original waveform plot
            waveform = voltage_data[y_click, x_click, :]
            waveform_plot.set_data(time_array*1e6, waveform)
            axs[0, 0].relim()
            axs[0, 0].autoscale_view()

            signal1 = isolated_signals_all[y_click, x_click, 0, :]
            signal2 = isolated_signals_all[y_click, x_click, 1, :]
            signal1_plot.set_data(time_array*1e6, signal1)
            signal2_plot.set_data(time_array*1e6, signal2 - 0.25)
            axs[0, 1].relim()
            axs[0, 1].autoscale_view()

            fig.canvas.draw()

fig.canvas.mpl_connect('button_press_event', on_click)
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
root.mainloop()


# In[ ]:




