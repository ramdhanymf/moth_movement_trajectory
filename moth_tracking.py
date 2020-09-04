# LIBRARY IMPORTS
from collections import deque
from imutils.video import VideoStream
import argparse
import datetime
import imutils
import time
import cv2
import numpy as np
import glob
import csv
#from matplotlib import pyplot as plt



# DEFINE ARGUMENTS
ap = argparse.ArgumentParser()
#ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
ap.add_argument("-t", "--tracker", type=str, default="kcf", help="OpenCV object tracker type")
ap.add_argument("-l", "--algo", type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
args = vars(ap.parse_args())

# selecting algorithm for background substraction
if args.get == 'MOG2':
    backSub = cv2.createBackgroundSubtractorCNT()
else:
    backSub = cv2.createBackgroundSubtractorKNN()

final_pts = deque(maxlen=None)
pts = deque(maxlen=None)
pos = 0

# object trackers algorithm
OPENCV_OBJECT_TRACKERS = {
		"csrt": cv2.TrackerCSRT_create,
		"kcf": cv2.TrackerKCF_create,
		"boosting": cv2.TrackerBoosting_create,
		"mil": cv2.TrackerMIL_create,
		"tld": cv2.TrackerTLD_create,
		"medianflow": cv2.TrackerMedianFlow_create,
		"mosse": cv2.TrackerMOSSE_create
	}

tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()

# initial condition
video_list = glob.glob(str(input("Please specify the directory contains all the footage >>> ")))
frame_space = 0 # give space from first noise



# LOOPING
for every_video in video_list:

    vs          = cv2.VideoCapture(every_video)
    frame = vs.read()[1]
    frame = imutils.resize(frame, width=500)

    # create white background image for trajectory
    white_image = np.zeros((frame.shape[0],frame.shape[1]), np.uint8)
    white_image.fill(255)
    white_image = cv2.cvtColor(white_image, cv2.COLOR_BGR2RGB)
    
    # select the bounding ROI
    initBB = cv2.selectROI("Footage", frame, fromCenter=False,showCrosshair=True)
    tracker.init(frame, initBB)

    # reset value of these variables
    frame_space = 0
    pts.clear()
    time.sleep(2.0)
    
    while every_video is not None:

        # get frame from video or webcam
        frame = vs.read()
        frame = frame[1]
        
        
        # initialize the if there is no frame detected in the video
        if frame is None:
            break
        
        frame = imutils.resize(frame, width=500)
        # update the frame and ROI box
        (success, box) = tracker.update(frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            # reset tracker variables
            initBB = None
            tracker.clear()
            tracker_new = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
            tracker = tracker_new
            
            # initialize again ROI 
            initBB = cv2.selectROI("Footage", frame, fromCenter=False,showCrosshair=True)
            tracker.init(frame, initBB)
            
            # udpate the tracker
            (success, box) = tracker.update(frame)
            (x, y, w, h) = [int(v) for v in box]
            box_center = (int((x+x+w)/2), int((y+y+h)/2))
            
            if final_pts is deque([]):
                final_pts = pts
            else:
                pts.append((0,0))
                pts.extend(final_pts)
                final_pts = pts
                pts.clear()
        
        elif key == ord("q"):
            # reset tracker variables
            initBB = None
            tracker.clear()
            tracker_new = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
            tracker = tracker_new
            success = False

        if success:
            (x, y, w, h) = [int(v) for v in box]

            # frame processing
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            fgMask = backSub.apply(frame)

            thresh = cv2.threshold(fgMask, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.erode(thresh, None, iterations=2)
            thresh = cv2.dilate(thresh, None, iterations=2)

            black = np.zeros((thresh.shape[0], thresh.shape[1]), np.uint8) #---black in RGB
            black1 = cv2.rectangle(black,(x,y),(x+w,y+h),(255, 255, 255), -1)   #---the dimension of the ROI
            
            fin = cv2.bitwise_and(black, thresh)

            cnts_2 = cv2.findContours(fin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts_2 = imutils.grab_contours(cnts_2)
            center = None

            for c in cnts_2:

                # determine the center of rectangle
                if len(c) > 0:
                    M = cv2.moments(c) # grabbing the area of rectangle
                    
                    # if moth is outside the region of interest continue looping
                    try:
                        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                        pts.appendleft(center)                 

                    except ZeroDivisionError:
                        continue
                    
                    # compute the bounding box
                    (x1, y1, w1, h1) = cv2.boundingRect(c)
                    cv2.rectangle(frame, (x1, y1), (x1+w1, y1+h1), (0, 255, 0), 1)

                # print the trajectory
                for i in range(1, len(pts)): 
                    if pts[i - 1] is None or pts[i] is None :
                        continue
                    else:
                        #cv2.line(frame, pts[i-1], pts[i], (0, 0, 255), thickness=1)
                        cv2.rectangle(frame, (x, y), (x + w, y + h),(0, 255, 255), 2)
                        cv2.line(white_image, pts[i-1], pts[i], (0, 0, 255), thickness=1)
                        

                cv2.imshow("Footage", frame)
                #cv2.imshow("FGMask", fgMask)
                #cv2.imshow("Maksed", fin)
                cv2.imshow("Trajectory", white_image)
                key = cv2.waitKey(1) & 0xFF

                # if the `q` key is pressed, break from the lop
                if key == ord("q"):
                    break

            

        else:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord("s"):
                # reset tracker variables
                initBB = None
                tracker.clear()
                tracker_new = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
                tracker = tracker_new
                
                # initialize again ROI 
                initBB = cv2.selectROI("Footage", frame, fromCenter=False,showCrosshair=True)
                tracker.init(frame, initBB)
                
                # udpate the tracker
                (success, box) = tracker.update(frame)
                (x, y, w, h) = [int(v) for v in box]
                box_center = (int((x+x+w)/2), int((y+y+h)/2))
                
                if final_pts is deque([]):
                    final_pts = pts
                else:
                    pts.append((0,0))
                    pts.extend(final_pts)
                    final_pts = pts
                    pts.clear()
                
            else:
                cv2.imshow("Footage", frame) 
                continue

    
    with open(str(every_video+'.csv'), 'w', newline='\n') as file:
        pts.extend(final_pts)
        final_pts = pts
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(["No", "X Axis", "Y Axis"])    
        for i in range(1, len(final_pts)):
            points = list()
            points = [i]
            for every_value in pts[len(final_pts)-i]:
                points.append(every_value)
            
            writer.writerow(points)

    trajectory_file_name = "/media/farhansamu21/86D2-4CAF/AASEC/final_footage/"+every_video+".png"
    cv2.imwrite(trajectory_file_name, white_image)

        
# CLEANUP OPEN WINDOWS
vs.stop() if video_list is None else vs.release()
cv2.destroyAllWindows()
