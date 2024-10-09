import cv2
import numpy as np
import time
import os
from utils import display_debug_windows, play_audio

class CameraProcessor:
    def __init__(self, video_src='0', debug=False):
        self.video_src = video_src
        self.debug = debug

        # Configuration
        self.src_width = 1920  # Input source width
        self.src_height = 1080  # Input source height
        self.scale_factor = 0.6  # Initial scaling to speed up algorithm
        self.rec_scale_factor = 0.3  # Scaling done to save video
        self.should_record = True  # Set to True to enable recording

        # Initialize variables
        self.frames_recorded = 0
        self.pos_array = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.consec_frames = 0
        self.consec_frames_needed = 50  # Number of frames needed to play sound
        self.start_time = time.time()  # Seed for playing sound clip
        self.clip_length = 40.0  # Length between sound clips in seconds
        self.out = None  # VideoWriter object

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
        self.roi_y0 = height // 2
        self.roi_y1 = height
        self.width = width

        # Convert to grayscale and crop
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = frame_gray[self.roi_y0:self.roi_y1, 0:self.width]

        return frame_gray

    def process_frame(self, frame_gray1, arduino_controller):
        ret, frame2 = self.cam.read()
        if not ret:
            return False, frame_gray1  # End of video stream

        # Downsample the frame
        frame2 = cv2.resize(frame2, (0, 0), fx=self.scale_factor, fy=self.scale_factor)

        # Crop the frame
        frame2 = frame2[self.roi_y0:self.roi_y1, 0:self.width]

        # Convert to grayscale
        frame_gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Further downsample for recording
        if self.should_record:
            subsample_orig = cv2.resize(
                frame2,  # Use the BGR frame
                (0, 0),
                fx=(1 / self.scale_factor) * self.rec_scale_factor,
                fy=(1 / self.scale_factor) * self.rec_scale_factor,
            )
        else:
            subsample_orig = None

        # Get difference image
        diff = cv2.absdiff(frame_gray1, frame_gray2)

        # Threshold the difference image
        thresh_val = 25
        maxValue = 255
        ret_thresh, thresh = cv2.threshold(diff, thresh_val, maxValue, cv2.THRESH_BINARY)

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
            self.update_tracking(best_column, avg_j, sub_w, frame2, arduino_controller)
        else:
            self.consec_frames = 0

        # Record the frame
        if self.should_record and subsample_orig is not None:
            if self.out is None:
                frame_height, frame_width = subsample_orig.shape[:2]
                frame_width = int(frame_width)
                frame_height = int(frame_height)
                self.initialize_video_writer(frame_width, frame_height)
            self.record_frame(subsample_orig)

        # Display debug windows if enabled
        if self.debug:
            display_debug_windows(frame2, diff, thresh, morphed, dilated)

        # Return the updated frame
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
        
    def remove_outliers(self, positions):
        """
        Removes outliers from the positions list based on standard deviation and returns the average position.

        Parameters:
        - positions: list of int, the list of servo positions.

        Returns:
        - pos: int, the average position after removing outliers.
        """
        import numpy as np

        # Convert the list to a NumPy array for convenience
        positions_array = np.array(positions)

        # Calculate the mean and standard deviation
        mean_pos = np.mean(positions_array)
        std_dev = np.std(positions_array)

        # Define a threshold for detecting outliers (e.g., 2 standard deviations)
        threshold = 2 * std_dev

        # Filter out positions that are beyond the threshold
        filtered_positions = positions_array[np.abs(positions_array - mean_pos) <= threshold]

        # If there are enough filtered positions, compute the average
        if len(filtered_positions) > 0:
            pos = int(np.mean(filtered_positions))
        else:
            # If all positions are outliers, default to the mean of the original positions
            pos = int(mean_pos)

        return pos


    def update_tracking(self, best_column, avg_j, sub_w, frame2, arduino_controller):
        # Draw lines if debug is enabled
        if self.debug:
            cv2.line(frame2, (best_column, 0), (best_column + 1, frame2.shape[0]), (0, 0, 255), 20)
            cv2.line(
                frame2,
                (best_column - avg_j, avg_j),
                (best_column + avg_j, avg_j),
                (0, 0, 255), 20,
            )

        # Map position for servo
        min_servo = 5
        max_servo = 175
        pos = min_servo + int(((best_column / 10.0) / sub_w) * (max_servo - min_servo))

        # Rolling average
        self.pos_array.append(pos)
        self.pos_array.pop(0)

        # Remove outliers
        pos = self.remove_outliers(self.pos_array)
        print("Servo Position: ", pos)

        # Send the position to the Arduino controller
        if arduino_controller is not None:
            arduino_controller.send_servo_position(pos)

        elapsed = time.time() - self.start_time  # Time since last clip played

        # Calculate central 10% horizontal bounds
        frame_width = frame2.shape[1]
        lower_bound = frame_width * 0.45
        upper_bound = frame_width * 0.55

        # Play sound if conditions are met
        self.consec_frames += 1
        if (elapsed > self.clip_length) and (lower_bound <= best_column <= upper_bound):
            print(f"Elapsed Time: {elapsed:.2f} seconds")
            print(f"Clip Length: {self.clip_length} seconds")
            self.consec_frames = 0
            play_audio()
            self.start_time = time.time()

    def initialize_video_writer(self, frame_width, frame_height):
        if not os.path.exists('recordings'):
            os.makedirs('recordings')
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Use 'XVID' for better compatibility
        self.out = cv2.VideoWriter(
            f"recordings/{int(time.time())}.avi",
            fourcc,
            15.0,  # fps
            (frame_width, frame_height),
        )
        if not self.out.isOpened():
            print("Failed to initialize VideoWriter.")
            self.out = None
        else:
            print(f"VideoWriter initialized with frame size: ({frame_width}, {frame_height})")

    def record_frame(self, frame):
        if self.out is None:
            print("Warning: VideoWriter is not initialized. Cannot write frame.")
            return

        self.out.write(frame)
        self.frames_recorded += 1

        # Start new video every X frames of recorded video
        if self.frames_recorded >= 10000:
            self.frames_recorded = 0
            self.out.release()  # Save video
            self.out = None  # It will be re-initialized in process_frame

    def release_resources(self):
        # Release resources
        self.cam.release()
        if self.out is not None:
            self.out.release()
        cv2.destroyAllWindows()
