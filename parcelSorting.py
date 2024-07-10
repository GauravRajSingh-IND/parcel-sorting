import cv2 as cv
import numpy as np
from function import maskFrame
from function import countObject

import matplotlib.pyplot as plt

# Window Name
winName = "Parcel Sorting"
cv.namedWindow(winName)

#Path of the video file.
path = "/Users/gauravsingh/Desktop/AI ENGINEER/Warehouse Sorting/Screen Recording 2024-07-09 at 8.04.45 PM.mov"

# Create video capture object.
cap = cv.VideoCapture(path)
if not cap.isOpened():
    print("issue while opening the video...")

# EXTRA PARAMETERS
###################
frameCount = 0

count_left = 0
status_left = 'low'

count_right = 0
status_right = 'low'
####################


# Background Subtractor Object.
bg_sub = cv.createBackgroundSubtractorKNN(history = 50)


# Main loop
while True:

    # Adding frameCount.
    frameCount += 1

    # Read frame one by one.
    has_frame, frame = cap.read()

    if not has_frame:
        print("No frame to read...")
        break

    # Create a mask from first frame.
    if frameCount == 1:
        mask = maskFrame(frame)

        plt.imshow(frame[:,:,::-1])
        plt.show()

    # Segment left and right belt area (foreground) from background.
    frame_fg = cv.bitwise_and(frame, frame, mask = mask)

    # Apply background subtractor to check the motion.
    frame_motion = bg_sub.apply(frame_fg)

    # Thresholding the result to make it binary.
    ret ,frame_motion_thresh = cv.threshold(frame_motion, 200, 255, cv.THRESH_BINARY)

    # apply erosion to remove small white pixel
    frame_erode = cv.erode(frame_motion_thresh, (5, 5))

    # Calculate percentage of white pixels in left and right belt area.
    per_l, per_r = countObject(frame_erode, frameCount)

    # Counting the objects on each belt by using below conditions.
    if status_left == 'low' and per_l >= 15:
        count_left += 1
        status_left = 'high'

    if status_left == 'high' and per_l <= 10:
        status_left = 'low'

    if status_right == 'low' and per_r >= 30:
        count_right += 1
        status_right = 'high'

    if status_right == 'high' and per_r <= 10:
        status_right = 'low'

    # Add Text on the frame to display the count.
    cv.putText(frame, f"Left: {count_left}", (18, 483), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0) , 3, cv.LINE_AA )
    cv.putText(frame, f"Right: {count_right}", (585, 483), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0) , 3, cv.LINE_AA )
    

    # Display the video.
    cv.imshow(winName,  frame)

    key = cv.waitKey(1)

    # Break the loop if user press 'q', 'Q' or esc key.
    if key == ord('q') or key == ord('Q') or key == 27:
        print("Stopped by user...")
        break

cap.release()
cv.destroyAllWindows()
    
