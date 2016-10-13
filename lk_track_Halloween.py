#!/usr/bin/env python

'''
Video Tracking App
====================

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
import video
from common import anorm2, draw_str
from time import clock
import math
from pprint import pprint
import serial
import time
#from scipy import ndimage

class App:
    def __init__(self, video_src):
        self.track_len = 10
        self.detect_interval = 5
        self.tracks = []

        self.frame_idx = 0

    def run(self, video_src):
        src_width = 1920 # input source
        src_heigth = 1080
        scale_factor = .6 # initial scaling to speed up algorithm
        rec_scale_factor = .3 # scaling done to save video
        self.cam = video.create_capture(video_src)
        self.cam.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, src_width)
        self.cam.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, src_heigth)
        
        # Configs
        debug = 1
        arduino_enabled = 1
        should_record = 1
        
        time.sleep(1)
        frames_recorded = 0
        pos_array = [0,0,0]
        ret, frame1 = self.cam.read()

        #capture first from
        frame1 = cv2.resize(frame1, (0,0), fx=scale_factor, fy=scale_factor)

        # Get heigth and width of downsampled image
        (height, width) = frame1.shape[:2]

        # Set the ROI for what we want to throw out
        roi_y0 = height/2
        roi_y1 = height

        # Convert to grey and crop
        frame_gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        frame_gray1 = frame_gray1[roi_y0:roi_y1, 0:width]

        # Saving videos of people reacting
        # Define the codec and create VideoWriter object
        if should_record:
            fourcc = cv2.cv.CV_FOURCC('I', 'Y', 'U', 'V')
            out = cv2.VideoWriter(str(time.time())+'.avi',
                                  fourcc,
                                  15.0, # fps
                                  (int(src_width*rec_scale_factor),int(src_heigth*rec_scale_factor*((roi_y1-roi_y0)/src_heigth))))

        #frame_gray1 = ndimage.rotate(frame_gray1, 180) # for if problem comes back

        # Establish serial connection with arduino if available
        try:
            ser = serial.Serial('COM5', 115200, timeout=0)
        except:
            arduino_enabled = 0
            print "Arduino not found on COM5."

        # Main while loop - you'll be here rest of the program
        while True:
            # Read the next frame
            ret, frame2 = self.cam.read()

            # Downsample just like you did the first frame
            frame2 = cv2.resize(frame2, (0,0), fx=scale_factor, fy=scale_factor)

            # Crop and make grey
            frame2 = frame2[roi_y0:roi_y1, 0:width]
            frame_gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            # Further downsample if recording
            if should_record:
                subsample_orig = cv2.resize(frame_gray2,
                                            (0,0),
                                            fx=(1/scale_factor)*rec_scale_factor,
                                            fy=(1/scale_factor)*rec_scale_factor)
            # Get difference image
            diff = cv2.absdiff(frame_gray1, frame_gray2)

            # Set threshold and maxValue
            thresh = 0
            maxValue = 255
 
            # Basic threshold example
            ret, thresh = cv2.threshold(diff, 25, maxValue, cv2.THRESH_BINARY);

            # Get rid of salt noise, then fill in gaps iteratively
            kernel = np.ones((int(8*scale_factor),int(8*scale_factor)),np.uint8)
            morphed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            dialated = cv2.dilate(morphed,kernel,iterations = 7)

            # One more round of downsampling for pixel by pixel search for moving objs
            subsample = cv2.resize(dialated, (0,0), fx=0.1, fy=0.1) 
            (sub_h, sub_w) = subsample.shape[:2]

            # Inititializations for mini-algorithm
            longest = 0
            best_column = 0
            longest_stretch = 0
            tracking = False
            j_start_tmp = 0
            j_start = 0
            j_end = 0

            # Search each column from top to bottom for moving areas.
            # Find the column that has the longest continuous strech of moving pixels
            for j in range(0,sub_w):
                for i in range(0,sub_h):
                    if subsample[i,j] > 0:
                        if tracking:
                            longest = longest + 1
                        else:
                            longest = 1
                            tracking = True
                            j_start_tmp = i
                    else:
                        if tracking:
                            if longest > longest_stretch:
                                longest_stretch = longest
                                best_column = j * 10 # 10 here is to offset .1 downsample
                                j_start = j_start_tmp * 10
                                j_end = i * 10
                            tracking = False

            if longest_stretch > 10*scale_factor:
                avg_j = int((j_end - j_start)/2) + j_start # find middle of brightest area for drawing

                # Draw the line
                if debug:
                    cv2.line(frame2, (best_column,0), (best_column + 1, 1080), (0,0,255), 20)
                    cv2.line(frame2, (best_column - avg_j,avg_j), (best_column + avg_j, avg_j), (0,0,255), 20)

                # Map to position for servo (servo goes from 0 to 90, but turns too far)
                min_servo = 10
                max_servo = 150

                # Use best col to map 0:1 across images to desired servo values
                pos = min_servo + int(((best_column/10.0)/sub_w)*(max_servo-min_servo))
                print "POS: "
                print pos

                # Use rolling average
                pos_array.append(pos)
                pos_array.pop(0)
                pos = int((pos_array[0] + pos_array[1] + pos_array[2])/3.0)
                print pos

                # Send angle to arduino
                if arduino_enabled:
                    ser.write(str(pos) + '\n')
                    time.sleep(0.01)
                    ret = ser.read()
                    print "and ret is "
                    print ret

                # Add frame to video output
                if should_record:
                    frames_recorded = frames_recorded + 1
                    # Start new video every X frames of recorded video
                    if frames_recorded > 500:
                        frames_recorded = 0
                        out.release() # Save video
                        del out
                        out = cv2.VideoWriter(str(time.time())+'.avi',
                                  fourcc,
                                  15.0,
                                  (int(src_width*rec_scale_factor),int(src_heigth*rec_scale_factor/2)))
                    # Write video to output buffer
                    out.write(subsample_orig)
                
            else:
                print "..."
            
            # open windows with original image, mask, res, and image with keypoints marked
            if debug:
                # Only bother showing images if in debug mode
                cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
                cv2.namedWindow('diff', cv2.WINDOW_NORMAL)
                cv2.namedWindow('thresh', cv2.WINDOW_NORMAL)
                cv2.namedWindow('morphed', cv2.WINDOW_NORMAL)
                cv2.namedWindow('dialated', cv2.WINDOW_NORMAL)
                cv2.imshow('diff', diff)
                cv2.imshow('thresh', thresh)
                cv2.imshow('morphed', morphed)
                cv2.imshow('dialated', dialated)  
                cv2.imshow('frame', frame2) 

            # Save this frame as last frame for difference image
            frame_gray1 = frame_gray2

            # Terminate program when user has focus and clicks escape
            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                if should_record:
                    # Save whatever video was being recorded
                    if frames_recorded > 1:
                        out.release()
                break

def main():
    import sys
    try: video_src = sys.argv[1]
    except: video_src = -1 # set this to 0 for built in or -1 for a menu-based choice

    print __doc__
    App(video_src).run(video_src)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
