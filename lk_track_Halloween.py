#!/usr/bin/env python3

'''
Video Tracking App
===================

Tracks random objects moving in the frame

Usage
-----
lk_track_Halloween.py [<video_source>]

Keys
----
ESC - exit
'''

import numpy as np
import cv2
import math
from pprint import pprint
import serial
import time
import subprocess

# from scipy import ndimage

class App:
    def __init__(self, video_src, debug=1):
        self.video_src = video_src
        self.debug = debug  # Debug flag
        self.track_len = 10
        self.detect_interval = 5
        self.tracks = []
        self.frame_idx = 0

        # Configuration
        self.src_width = 1920  # Input source width
        self.src_height = 1080  # Input source height
        self.scale_factor = 1  # Initial scaling to speed up algorithm
        self.rec_scale_factor = 1  # Scaling done to save video
        self.arduino_enabled = True
        self.should_record = True

        # Initialize variables
        self.frames_recorded = 0
        self.pos_array = [0, 0, 0]
        self.consec_frames = 0
        self.consec_frames_needed = 50  # Number of frames needed to play sound
        self.start_time = time.time()  # Seed for playing Halloween theme
        self.clip_length = 40.0  # Length of theme in seconds

    def initialize_camera(self):
        self.cam = cv2.VideoCapture(self.video_src)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.src_width)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.src_height)
        time.sleep(1)

    def capture_initial_frame(self):
        ret, frame = self.cam.read()
        if not ret:
            print("Failed to capture initial frame from camera.")
            return None
        frame = cv2.resize(frame, (0, 0), fx=self.scale_factor, fy=self.scale_factor)
        return frame

    def preprocess_frame(self, frame):
        # Get height and width of downsampled image
        (height, width) = frame.shape[:2]

        # Set the ROI for what we want to throw out
        roi_y0 = height // 2
        roi_y1 = height

        # Convert to grayscale and crop
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = frame_gray[roi_y0:roi_y1, 0:width]

        return frame_gray, (roi_y0, roi_y1, width)

    def initialize_video_writer(self, roi_y0, roi_y1):
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'IYUV')
        out = cv2.VideoWriter(
            f"{time.time()}.avi",
            fourcc,
            15.0,  # fps
            (
                int(self.src_width * self.rec_scale_factor),
                int(self.src_height * self.rec_scale_factor * ((roi_y1 - roi_y0) / self.src_height)),
            ),
        )
        return out

    def initialize_serial_connection(self):
        try:
            ser = serial.Serial('COM5', 115200, timeout=0)
            print("Arduino connected on COM5.")
        except serial.SerialException:
            self.arduino_enabled = False
            ser = None
            print("Arduino not found on COM5.")
        return ser

    def process_frame(self, frame_gray1, roi_y0, roi_y1, width, ser, out):
        ret, frame2 = self.cam.read()
        if not ret:
            return False, frame_gray1  # End of video stream

        # Downsample the frame
        frame2 = cv2.resize(frame2, (0, 0), fx=self.scale_factor, fy=self.scale_factor)

        # Crop and convert to grayscale
        frame2 = frame2[roi_y0:roi_y1, 0:width]
        frame_gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Further downsample if recording
        if self.should_record:
            subsample_orig = cv2.resize(
                frame_gray2,
                (0, 0),
                fx=(1 / self.scale_factor) * self.rec_scale_factor,
                fy=(1 / self.scale_factor) * self.rec_scale_factor,
            )
        else:
            subsample_orig = None

        # Get difference image
        diff = cv2.absdiff(frame_gray1, frame_gray2)

        # Threshold the difference image
        ret_thresh, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        # Morphological operations
        kernel_size = int(8 * self.scale_factor)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        morphed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        dilated = cv2.dilate(morphed, kernel, iterations=7)

        # Downsample for pixel-by-pixel search
        subsample = cv2.resize(dilated, (0, 0), fx=0.1, fy=0.1)
        (sub_h, sub_w) = subsample.shape[:2]

        # Find the column with the longest continuous stretch of moving pixels
        best_column, avg_j = self.find_best_column(subsample, sub_h, sub_w)

        if best_column is not None:
            self.update_tracking(best_column, avg_j, sub_w, frame2, ser)
            if self.should_record:
                self.record_frame(out, subsample_orig)
        else:
            self.consec_frames = 0
            print("...")  # No significant movement detected

        # Display debug windows if enabled
        if self.debug:
            self.display_debug_windows(frame2, diff, thresh, morphed, dilated)

        return True, frame_gray2

    def find_best_column(self, subsample, sub_h, sub_w):
        longest_stretch = 0
        best_column = None
        avg_j = 0

        for j in range(sub_w):
            longest = 0
            tracking = False
            j_start_tmp = 0
            j_start = 0
            j_end = 0
            for i in range(sub_h):
                if subsample[i, j] > 0:
                    if tracking:
                        longest += 1
                    else:
                        longest = 1
                        tracking = True
                        j_start_tmp = i
                else:
                    if tracking:
                        if longest > longest_stretch:
                            longest_stretch = longest
                            best_column = j * 10  # Offset due to downsampling
                            j_start = j_start_tmp * 10
                            j_end = i * 10
                        tracking = False
            if tracking and longest > longest_stretch:
                longest_stretch = longest
                best_column = j * 10
                j_start = j_start_tmp * 10
                j_end = sub_h * 10

        if longest_stretch > 7 * self.scale_factor:
            avg_j = int((j_end - j_start) / 2) + j_start  # Middle of brightest area
            return best_column, avg_j
        else:
            return None, None

    def update_tracking(self, best_column, avg_j, sub_w, frame2, ser):
        # Draw lines if debug is enabled
        if self.debug:
            cv2.line(frame2, (best_column, 0), (best_column + 1, 1080), (0, 0, 255), 20)
            cv2.line(
                frame2,
                (best_column - avg_j, avg_j),
                (best_column + avg_j, avg_j),
                (0, 0, 255),
                20,
            )

        # Map position for servo
        min_servo = 10
        max_servo = 150
        pos = min_servo + int(((best_column / 10.0) / sub_w) * (max_servo - min_servo))
        print("POS:")
        print(pos)

        # Rolling average
        self.pos_array.append(pos)
        self.pos_array.pop(0)
        pos = int(sum(self.pos_array) / len(self.pos_array))
        print(pos)

        # Send angle to Arduino
        if self.arduino_enabled and ser is not None:
            ser.write((str(pos) + '\n').encode())
            time.sleep(0.01)
            ser.read()

        elapsed = time.time() - self.start_time  # Time since last clip played

        # Play sound if conditions are met
        self.consec_frames += 1
        if (elapsed > self.clip_length) and (85 < pos < 105):
            print(elapsed)
            print(self.clip_length)
            self.consec_frames = 0
            subprocess.Popen(["python", "play_sound.py"])
            self.start_time = time.time()

    def record_frame(self, out, subsample_orig):
        self.frames_recorded += 1
        # Start new video every X frames of recorded video
        if self.frames_recorded > 1000:
            self.frames_recorded = 0
            out.release()  # Save video
            del out
            out = self.initialize_video_writer()
        out.write(subsample_orig)

    def display_debug_windows(self, frame2, diff, thresh, morphed, dilated):
        # Helper function to convert images to BGR
        def to_bgr(img):
            if len(img.shape) == 2:
                return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                return img

        # Helper function to resize images while maintaining aspect ratio
        def resize_with_aspect_ratio(image, max_width=None, max_height=None):
            (h, w) = image.shape[:2]
            if max_width is not None and w > max_width:
                scale = max_width / w
                new_w = int(w * scale)
                new_h = int(h * scale)
            elif max_height is not None and h > max_height:
                scale = max_height / h
                new_w = int(w * scale)
                new_h = int(h * scale)
            else:
                new_w, new_h = w, h
            return cv2.resize(image, (new_w, new_h))

        # Helper function to create a label bar above each image
        def create_label_bar(text, width):
            label_height = 40  # Height of the label bar
            label_bar = np.full((label_height, width, 3), border_color, dtype=np.uint8)  # Grey background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.5  # Increased font size for larger images
            font_thickness = 3
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            text_x = (width - text_size[0]) // 2
            text_y = (label_height + text_size[1]) // 2 - 5  # Adjust vertical position
            cv2.putText(label_bar, text, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness)
            return label_bar

        # Set the border color to grey
        border_color = [127, 127, 127]  # Grey color
        border_size = 20  # Increased border size for better visibility

        # Convert images to BGR
        frame2_bgr = to_bgr(frame2)
        diff_bgr = to_bgr(diff)
        thresh_bgr = to_bgr(thresh)
        morphed_bgr = to_bgr(morphed)
        dilated_bgr = to_bgr(dilated)

        # Resize images while keeping aspect ratio
        max_width = 720  # Increased size (3X larger)
        frame2_bgr = resize_with_aspect_ratio(frame2_bgr, max_width=max_width)
        diff_bgr = resize_with_aspect_ratio(diff_bgr, max_width=max_width)
        thresh_bgr = resize_with_aspect_ratio(thresh_bgr, max_width=max_width)
        morphed_bgr = resize_with_aspect_ratio(morphed_bgr, max_width=max_width)
        dilated_bgr = resize_with_aspect_ratio(dilated_bgr, max_width=max_width)

        # Create label bars
        frame2_label = create_label_bar('Frame', frame2_bgr.shape[1])
        diff_label = create_label_bar('Difference', diff_bgr.shape[1])
        thresh_label = create_label_bar('Threshold', thresh_bgr.shape[1])
        morphed_label = create_label_bar('Morphed', morphed_bgr.shape[1])
        dilated_label = create_label_bar('Dilated', dilated_bgr.shape[1])

        # Create horizontal border between label and image
        label_image_border = np.full((border_size, frame2_bgr.shape[1], 3), border_color, dtype=np.uint8)

        # Stack label bars above images with horizontal border
        frame2_with_label = np.vstack((frame2_label, label_image_border, frame2_bgr))
        diff_with_label = np.vstack((diff_label, label_image_border, diff_bgr))
        thresh_with_label = np.vstack((thresh_label, label_image_border, thresh_bgr))
        morphed_with_label = np.vstack((morphed_label, label_image_border, morphed_bgr))
        dilated_with_label = np.vstack((dilated_label, label_image_border, dilated_bgr))

        # Add border around each image
        images = [frame2_with_label, diff_with_label, thresh_with_label, morphed_with_label, dilated_with_label]
        images_with_borders = []
        for img in images:
            img_with_border = cv2.copyMakeBorder(
                img,
                border_size,  # Top
                border_size,  # Bottom
                border_size,  # Left
                border_size,  # Right
                cv2.BORDER_CONSTANT,
                value=border_color,
            )
            images_with_borders.append(img_with_border)

        images = images_with_borders

        # Find maximum image dimensions
        max_height = max(img.shape[0] for img in images)
        max_width = max(img.shape[1] for img in images)

        # Pad images to have the same dimensions
        for i in range(len(images)):
            img = images[i]
            h, w = img.shape[:2]
            top = (max_height - h) // 2
            bottom = max_height - h - top
            left = (max_width - w) // 2
            right = max_width - w - left
            images[i] = cv2.copyMakeBorder(
                img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=border_color
            )

        # Create an empty image for the missing sixth slot
        empty_image = np.full((max_height, max_width, 3), border_color, dtype=np.uint8)

        # Arrange images in a grid without additional borders (since each image already has borders)
        top_row = np.hstack((images[0], images[1], images[2]))
        bottom_row = np.hstack((images[3], images[4], empty_image))

        # Create horizontal border between top and bottom rows
        horizontal_border_between_rows = np.full(
            (border_size, top_row.shape[1], 3), border_color, dtype=np.uint8
        )

        # Stack the rows vertically with a horizontal border
        combined_image = np.vstack((top_row, horizontal_border_between_rows, bottom_row))

        # Display the combined image
        cv2.imshow('Debug Window', combined_image)



    def handle_exit(self, out):
        # Terminate program when user presses ESC
        ch = cv2.waitKey(1) & 0xFF
        if ch == 27:
            if self.should_record and self.frames_recorded > 1:
                out.release()
            return True
        return False

    def run(self):
        self.initialize_camera()
        frame1 = self.capture_initial_frame()
        if frame1 is None:
            return

        frame_gray1, roi_params = self.preprocess_frame(frame1)
        roi_y0, roi_y1, width = roi_params

        if self.should_record:
            out = self.initialize_video_writer(roi_y0, roi_y1)
        else:
            out = None

        ser = self.initialize_serial_connection()

        while True:
            success, frame_gray1 = self.process_frame(frame_gray1, roi_y0, roi_y1, width, ser, out)
            if not success:
                break
            if self.handle_exit(out):
                break

def main():
    import sys
    try:
        video_src_arg = sys.argv[1]
    except IndexError:
        video_src_arg = '0'  # Default to '0' for the built-in webcam

    print(__doc__)

    # Try to convert video_src_arg to an integer (camera index)
    try:
        video_src = int(video_src_arg)
    except ValueError:
        video_src = video_src_arg  # Use as is (e.g., a filename)

    debug = 1  # Set to 0 to disable debug mode

    app = App(video_src, debug=debug)
    app.run()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
