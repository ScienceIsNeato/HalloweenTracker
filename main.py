#!/usr/bin/env python3

'''
slyaborham_lincoln - Video Tracking App
=======================================

Tracks moving objects in the frame, controls a servo motor via Arduino, and plays spooky sounds when motion is detected.

Usage
-----
main.py [--video_src <video_source>] [--serial_port <port>] [--debug]

Arguments
---------
--video_src <video_source>
    Specify the video source. It can be the camera index (e.g., '0', '1') or a video file path. Defaults to '0' if not provided.

--serial_port <port>
    Specify the serial port for the Arduino connection (e.g., 'COM3', 'COM5', '/dev/ttyUSB0'). Defaults to 'COM5' if not provided.

--debug
    Enable debug mode to display processing windows and additional output.

--camera_fov <degrees>
    Specify the camera's horizontal field of view in degrees. Defaults to 180.0 if not provided.

--camera_upside_down
    Specify if the camera is mounted upside down. This will flip the camera feed vertically.

Keys
----
ESC - exit the program

Description
-----------
This script captures video from the specified source, processes each frame to detect motion, and tracks the position of moving objects. When motion is detected:

- Controls a servo motor connected to an Arduino, pointing towards the movement.
- Plays a random spooky sound from the 'sound_clips' directory.
- Records video clips of the motion events and saves them in the 'recordings' folder.

'''

import argparse
import sys
import time
from camera import CameraProcessor
from arduino import ArduinoController
from utils import handle_exit

def main():
    parser = argparse.ArgumentParser(description='slyaborham_lincoln - Video Tracking App')
    parser.add_argument('--video_src', default='0', help='Video source (default: 0)')
    parser.add_argument('--serial_port', default='COM5', help='Serial port for Arduino (default: COM5)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--camera_fov', type=float, default=180.0, help='Camera field of view in degrees (default: 180.0)')
    parser.add_argument('--camera_upside_down', action='store_true', help='Camera is mounted upside down')

    args = parser.parse_args()

    print(__doc__)

    # Try to convert video_src to an integer (camera index)
    try:
        video_src = int(args.video_src)
    except ValueError:
        video_src = args.video_src  # Use as is (e.g., a filename)

    # Initialize camera processor
    print("Initializing camera with an fov of {} degrees...".format(args.camera_fov))
    camera_processor = CameraProcessor(video_src=video_src, debug=args.debug, camera_fov=args.camera_fov, is_upside_down=args.camera_upside_down)

    # Initialize Arduino controller
    arduino_controller = ArduinoController(serial_port=args.serial_port)

    # Start video capture
    camera_processor.initialize_camera()
    frame1 = camera_processor.capture_initial_frame()
    if frame1 is None:
        return

    frame_gray1 = camera_processor.preprocess_frame(frame1)

    # Main loop
    try:
        while True:
            success, frame_gray1 = camera_processor.process_frame(frame_gray1, arduino_controller)
            if not success:
                break
            if handle_exit():
                break
    finally:
        # Clean up resources
        camera_processor.release_resources()
        arduino_controller.close()
        print("Application terminated.")

if __name__ == '__main__':
    main()
