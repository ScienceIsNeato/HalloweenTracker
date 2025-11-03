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
python main.py [--video_src <video_source>] [--serial_port <port>] [--debug] [--camera_fov <degrees>] [--camera_upside_down]
```

### Command-Line Arguments:

- `--video_src <video_source>`: Specify the video source. It can be the camera index (e.g., 0, 1) or a video file path. Defaults to 0 if not provided.

- `--serial_port <port>`: Specify the serial port for the Arduino connection (e.g., COM3, COM5, /dev/ttyUSB0). Defaults to COM5 if not provided.

- `--debug`: Enable debug mode to display processing windows and additional output.

- `--camera_fov <degrees>`: Specify the camera's horizontal field of view in degrees. Defaults to 180.0 if not provided. This adjusts the servo's range of motion to match your camera's field of view.

- `--camera_upside_down`: Specify if the camera is mounted upside down. This will flip the camera feed vertically.

### Examples:

- Use the default camera and serial port:
```bash
python main.py
```

- Specify a different camera index:
```bash
python main.py --video_src 1
```

- Specify a video file as the source:
```bash
python main.py --video_src path/to/video.mp4
```

- Specify a different serial port:
```bash
python main.py --serial_port COM3
```

- Enable debug mode:
```bash
python main.py --debug
```

- Set camera field of view (e.g., for a 120-degree camera):
```bash
python main.py --camera_fov 120.0
```

- Use an upside-down mounted camera:
```bash
python main.py --camera_upside_down
```

- Combine multiple arguments:
```bash
python main.py --video_src 1 --serial_port COM3 --debug --camera_fov 120.0
```

## Arduino Setup:

* Upload the Meyers.ino sketch to your Arduino.
* Ensure the Arduino is connected to the correct serial port (COM5 in the script; adjust if necessary).

## Notes

* **Motion Detection**: If you sit perfectly still, the display windows may appear static. Move around to see the motion tracking in action.
* **Sound Clips**: Place your audio files (`.wav`, `.mp3`, `.ogg`, `.flac`) in the `sound_clips` directory. The system will play random clips from this directory when motion is detected in the center of the frame.
* **Serial Communication**: Adjust the serial port using the `--serial_port` argument to match your system's configuration. The program will list available ports if the specified one isn't found.
* **Recording Videos**: The script continuously records video clips of motion events and saves them in the `recordings` folder as `.avi` files.
* **Debug Mode**: Enabling debug mode will display processing windows showing the frame, difference, threshold, morphed, and dilated images in a grid layout.
* **Camera Field of View**: If your camera doesn't have a 180-degree field of view, use the `--camera_fov` argument to set the correct angle. This ensures the servo's movement range matches what the camera can actually see.

## Project Structure

* **main.py**: Entry point for the application. Handles command-line arguments and coordinates the camera and Arduino components.
* **camera.py**: Contains the `CameraProcessor` class that handles video capture, motion detection, and recording.
* **arduino.py**: Contains the `ArduinoController` class that manages serial communication with the Arduino.
* **utils.py**: Utility functions including the debug display window renderer and other helper functions.
* **play_audio.py**: Contains the `SoundPlayer` class that plays random sound clips with thread-safe playback management.
* **Meyers/Meyers.ino**: Arduino sketch for controlling the servo motor.
* **requirements.txt**: Python dependencies list.
* **sound_clips/**: Directory containing audio files to be played.
* **recordings/**: Directory where recorded video clips are saved.

## License

This project is open-source and available for use and modification.