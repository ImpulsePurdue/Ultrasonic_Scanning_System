# Custom Ultrasonic Immersion System

## Overview
This repository contains CAD drawings, firmware, and data processing scripts for a **custom-built ultrasonic immersion scanning system**, modified from a Voron 3D printer. The system enables precise ultrasonic scanning for nondestructive evaluation (NDE) applications, particularly in materials research.

## Project Features
- **Hardware:** Modified Voron 3D printer frame, custom transducer attachment, motion system powered by stepper motors.
- **Firmware:** Runs on **Marlin**, allowing coordinate-based scanning.
- **Software & Data Processing:**
  - Python scripts for controlling the scanner and logging ultrasonic data.
  - CSV-based data collection from oscilloscope readings.
  - Post-processing for **C-scan visualization** and **wavespeed calculations**.

## Repository Structure
## Installation & Usage
### 1. Setting up the scanner
- Assemble the frame and ensure the **XY motion system** is calibrated.
- Attach the **transducer** with adjustable positioning.
- Seal the immersion tank properly to prevent leaks.

### 2. Running the system
- Upload **Marlin firmware** to the motion controller.
- Use the provided Python scripts to define **scan coordinates**.
- Collect ultrasonic pulse data and save it as **CSV files**.

### 3. Post-Processing
- Use `wavespeed_analysis.py` to compute material properties.
- Generate **C-scan images** using `cscan_visualization.py`.

## Supplementary Data for Paper
This repository is supplementary to our paper:  
📄 **[Paper Title]** *(Link to published paper or DOI)*  

If you use this work in your research, please cite it as:  
```bibtex
@article{YourName2024,
  author  = {Harshith ...},
  title   = {Title of your paper},
  journal = {NDT&E International},
  year    = {2024},
  doi     = {DOI link}
}
