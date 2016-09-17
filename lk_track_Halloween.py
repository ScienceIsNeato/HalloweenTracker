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

        while True:
            ret, frame2 = self.cam.read()
            vis = frame2.copy
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

            # Set up the SimpleBlobdetector with default parameters.
            params = cv2.SimpleBlobDetector_Params()
             
            # Change thresholds
            params.minThreshold = 0;
            params.maxThreshold = 256;
             
            # Filter by Area.
            params.filterByArea = True
            params.minArea = 10
             
            # Filter by Circularity
            params.filterByCircularity = False
            params.minCircularity = 0.1
             
            # Filter by Convexity
            params.filterByConvexity = False
            params.minConvexity = 0.5
             
            # Filter by Inertia
            params.filterByInertia =False
            params.minInertiaRatio = 0.5
             
            detector = cv2.SimpleBlobDetector(params)
         
            # Detect blobs.
            reversemask=255-dialated
            keypoints = detector.detect(reversemask)

            blob_x = 0
            
            if keypoints:
                print "found %d blobs" % len(keypoints)
                #if len(keypoints) > 4:
                    # if more than four blobs, keep the four largest
                keypoints.sort(key=(lambda s: s.size))
                    #keypoints=keypoints[0:3]
                point = keypoints[0].pt
                print "point is"
                pprint(point)
                blob_x = int(point[0])
                
            else:
                print "no blobs"
         
            # Draw green circles around detected blobs
            im_with_keypoints = cv2.drawKeypoints(frame2, keypoints, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            if keypoints:
                cv2.line(im_with_keypoints, (blob_x,0), (blob_x, 1000), (0,0,255), 14)
                
            # open windows with original image, mask, res, and image with keypoints marked
            cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
            cv2.namedWindow('blobs', cv2.WINDOW_NORMAL)
            cv2.namedWindow('diff', cv2.WINDOW_NORMAL)
            cv2.namedWindow('thresh', cv2.WINDOW_NORMAL)
            cv2.namedWindow('morphed', cv2.WINDOW_NORMAL)
            cv2.namedWindow('dialated', cv2.WINDOW_NORMAL)
            cv2.imshow('diff', diff)
            cv2.imshow('thresh', thresh)
            cv2.imshow('morphed', morphed)
            cv2.imshow('dialated', dialated)  
            cv2.imshow('frame',thresh)
            cv2.imshow('blobs', im_with_keypoints)            

            frame_gray1 = frame_gray2
            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                break
            
            #vis = frame2.copy()

            if 0: #len(self.tracks) > 0:
                img0, img1 = self.prev_gray, frame_gray1
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                d = abs(p0-p0r).reshape(-1, 2).max(-1)
                good = d < 0.1
                new_tracks = []
                longest = 0
                tmpx = 0
                tmpy = 0
                for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                    if not good_flag:
                        continue
                    tr.append((x, y))
                    if len(tr) > self.track_len:
                        del tr[0]
                    dist = math.pow(tr[0][0]-tr[len(tr)-1][0],2) + math.pow(tr[0][1]-tr[len(tr)-1][1],2)
                    if dist < 10:
                        continue
                    new_tracks.append(tr)
                    if dist > longest:
                        tmpx = x
                        tmpy = y
                        longest = dist
                    cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
                self.tracks = new_tracks
                cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
                cv2.circle(vis, (tmpx, tmpy), 2, (0, 0, 255), 4)
                cv2.line(vis, (tmpx,0), (tmpx, 1000), (0,0,255))

                draw_str(vis, (20, 20), 'track count: %d' % len(self.tracks))

            if 0: #self.frame_idx % self.detect_interval == 0:
                mask = np.zeros_like(frame_gray1)
                mask[:] = 255
                for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                    cv2.circle(mask, (x, y), 5, 0, -1)
                p = cv2.goodFeaturesToTrack(frame_gray1, mask = mask, **feature_params)
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        self.tracks.append([(x, y)])


            # self.frame_idx += 1
            # self.prev_gray = frame_gray1
            # cv2.namedWindow("lk_track", cv2.WINDOW_NORMAL)
            # cv2.imshow('lk_track', diff)

           

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
