#!/usr/bin/env python

'''
Lucas-Kanade tracker
====================

Lucas-Kanade sparse optical flow demo. Uses goodFeaturesToTrack
for track initialization and back-tracking for match verification
between frames.

Usage
-----
lk_track.py [<video_source>]


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
import time


lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.001, # changed from .1
                       minDistance = 7,
                       blockSize = 7 )

class App:
    def __init__(self, video_src):
        self.track_len = 10
        self.detect_interval = 5
        self.tracks = []
        self.cam = video.create_capture(video_src)
        
        self.cam.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 1920)
        self.cam.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 1080)
        self.frame_idx = 0
        global getLongestLine



    def run(self):
        time.sleep(1)
        ret, frame1 = self.cam.read()
        frame_gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        rand = 0
        while True:
            ret, frame2 = self.cam.read()
            frame_gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(frame_gray1, frame_gray2)

            # Set threshold and maxValue
            thresh = 0
            maxValue = 255
 
            # Basic threshold example
            ret, thresh = cv2.threshold(diff, 25, maxValue, cv2.THRESH_BINARY);

            kernel = np.ones((10,10),np.uint8)
            morphed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            dialated = cv2.dilate(morphed,kernel,iterations = 5)

            (_h, _w) = dialated.shape[:2]
            
            subsample = cv2.resize(dialated, (0,0), fx=0.1, fy=0.1) 

            (sub_h, sub_w) = subsample.shape[:2]

            longest = 0
            best_column = 0
            longest_stretch = 0
            tracking = False
            j_start_tmp = 0
            j_start = 0
            j_end = 0
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
                                best_column = j * 10
                                j_start = j_start_tmp * 10
                                j_end = i * 10
                            tracking = False

            if longest_stretch > 10:
                avg_j = int((j_end - j_start)/2) + j_start
                cv2.line(frame2, (best_column,0), (best_column + 1, 1080), (0,0,255), 20)
                cv2.line(frame2, (best_column - avg_j,avg_j), (best_column + avg_j, avg_j), (0,0,255), 20)
                print "longest stretch is ", longest_stretch
                print "at column ", best_column
            else:
                print "..."
            
            # open windows with original image, mask, res, and image with keypoints marked
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

            frame_gray1 = frame_gray2
            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                break
            
            

    def getLongestLine(self):
        global longest
        longest = 0
        for tr in self.tracks:
            dist = math.pow(tr[0][0]-tr[0][1],2) + math.pow(tr[-1][1]-tr[-1][1],2)
            if dist > longest:
                bestPair = (tr)

def main():
    import sys
    try: video_src = sys.argv[1]
    except: video_src = -1 # set this to 0 for built in

    print __doc__
    App(video_src).run()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
