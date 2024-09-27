# Halloween Motion Tracker

A Python application that uses a webcam to detect motion and control a servo motor via Arduino. When motion is detected, the servo points towards the movement, and a random spooky sound plays.

## Features

- Tracks movement using a connected camera.
- Controls a servo motor connected to an Arduino based on detected motion.
- Plays random sound clips when motion is detected.

## Requirements

- **Python Version**: Python 3.x
- **Dependencies**: Install using `requirements.txt`
  ```bash
  pip install -r requirements.txt
  ```

## Hardware Setup
* Camera: Built-in or external webcam.
* Arduino: Connected via serial port.
* Servo Motor: Connected to Arduino (analog pin A0).

## Usage
* Determine Camera Index:

  * Use the following script to identify your camera index:

```python
import cv2

for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera index {i} is available.")
        cap.release()
    else:
        print(f"Camera index {i} is not available.")
```

* Run the script:
```bash
python lk_track_Halloween.py [camera_index]
```
* Replace [camera_index] with the index number identified.

## Arduino Setup:

* Upload the Meyers.ino sketch to your Arduino.
* Ensure the Arduino is connected to the correct serial port (COM5 in the script; adjust if necessary).

## Notes

* Motion Detection: If you sit perfectly still, the display windows may appear black. Move around to see the motion tracking in action.
* Sound Clips: Place your .wav audio files in the sound_clips directory.
* Serial Communication: Adjust the serial port in the script (COM5) to match your system.

## Files

* lk_track_Halloween.py: Main Python script for motion tracking.
* play_audio.py: Plays random sound clips when motion is detected.
* Meyers.ino: Arduino sketch for controlling the servo motor.
* requirements.txt: Python dependencies list.


## License
* This project is open-source and available for use and modification.

css
Copy code

You can copy and paste the above content into Visual Studio Code, and the formatting should be preserved.