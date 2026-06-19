# Real-Time Driver Drowsiness Detection

## Overview

Real-Time Driver Drowsiness Detection is a computer vision-based system that monitors a driver's eye movements using a webcam and facial landmark detection. The system calculates the Eye Aspect Ratio (EAR) to detect signs of fatigue and drowsiness. When drowsiness is detected, an alert is triggered to improve driver awareness and road safety.

## Features

* Real-time eye tracking using a webcam
* Facial landmark detection with MediaPipe
* Eye Aspect Ratio (EAR) calculation
* Drowsiness detection and alert system
* User-friendly web interface
* Live monitoring and status updates

## Technologies Used

* Python
* Flask
* OpenCV
* MediaPipe
* HTML
* CSS

## Project Structure

```text
├── app.py
├── templates/
├── static/
├── face_landmarker.task
├── dashboard.html
├── dashboard.css
└── README.md
```

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/Real-Time-Driver-Drowsiness-Detection.git
   ```

2. Navigate to the project folder:

   ```bash
   cd Real-Time-Driver-Drowsiness-Detection
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:

   ```bash
   python app.py
   ```

## Usage

* Start the application.
* Allow webcam access.
* The system will monitor eye movements in real time.
* An alert will be generated when drowsiness is detected.

## Future Enhancements

* Mobile application integration
* Advanced AI-based fatigue analysis
* GPS-based emergency notifications
* Cloud-based monitoring and reporting

## Author

**SreyaShinu-21**
