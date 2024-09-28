# Halloween Motion Tracker

A Python application that uses a webcam to detect motion and control a servo motor via Arduino. When motion is detected, the servo points towards the movement, and a random spooky sound plays.

## Features

- Tracks movement using a connected camera.
- Controls a servo motor connected to an Arduino based on detected motion.
- Plays random sound clips when motion is detected.
- Records video clips of detected motion events.

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
python lk_track_Halloween.py [--video_src <video_source>] [--serial_port <port>] [--debug]
```

### Command-Line Arguments:

- --video_src <video_source>: Specify the video source. It can be the camera index (e.g., 0, 1) or a video file path. Defaults to 0 if not provided.

- --serial_port <port>: Specify the serial port for the Arduino connection (e.g., COM3, COM5, /dev/ttyUSB0). Defaults to COM5 if not provided.

- --debug: Enable debug mode to display processing windows and additional output.

### Examples:

- Use the default camera and serial port:
```
python lk_track_Halloween.py
```
- Specify a different camera index:
```
python lk_track_Halloween.py --video_src 1
```
- Specify a video file as the source:
```
python lk_track_Halloween.py --video_src path/to/video.mp4
```
- Specify a different serial port:
```
python lk_track_Halloween.py --serial_port COM3
```
- Enable debug mode:
```
python lk_track_Halloween.py --debug
```
- Combine arguments:
```
python lk_track_Halloween.py --video_src 1 --serial_port COM3 --debug
```

## Arduino Setup:

* Upload the Meyers.ino sketch to your Arduino.
* Ensure the Arduino is connected to the correct serial port (COM5 in the script; adjust if necessary).

## Notes

* Motion Detection: If you sit perfectly still, the display windows may appear static. Move around to see the motion tracking in action.
* Sound Clips: Place your .wav audio files in the sound_clips directory. The play_audio.py script will play random clips from this directory when motion is detected.
* Serial Communication: Adjust the serial port using the --serial_port argument to match your system's configuration.
* Recording Videos: The script records video clips of motion events and saves them in the recordings folder.
* Debug Mode: Enabling debug mode will display processing windows showing the frame, difference, threshold, morphed, and dilated images.

## Files

* lk_track_Halloween.py: Main Python script for motion tracking.
* play_audio.py: Plays random sound clips when motion is detected.
* Meyers.ino: Arduino sketch for controlling the servo motor.
* requirements.txt: Python dependencies list.
* sound_clips/: Directory containing audio files to be played.
* recordings/: Directory where recorded video clips are saved.


## License
* This project is open-source and available for use and modification.

css
Copy code

You can copy and paste the above content into Visual Studio Code, and the formatting should be preserved.