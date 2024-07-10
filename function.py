import cv2 as cv
import numpy as np

# create a mask for left and right belt
def maskFrame(frame):

    # Create a black image of same size as frame.
    mask1 = np.zeros_like(frame)

    # Use one channel as we just required 1 channel for masking
    mask1 = mask1[:,:,0]

    # Add white pixels to the required area.
    mask1[600:700 ,78:217] = 255
    mask1[558:637, 527:668] = 255
    mask1[354:373, 74:103] = 255

    return mask1

# Count Non Zero Pixela in both Belt.
def countObject(frame, frameCount):

    # Crop left belt and right brlt area.
    area_left = frame[600:700 ,78:217]
    area_right = frame[558:637, 527:668]

    if frameCount >= 1:
        # Calculate total area of left belt.
        wl, hl = area_left.shape
        area_l = int(wl * hl)

        # Calculate total area of right belt.
        wr, hr = area_right.shape
        area_r = int(wr * hr)

    # Counting Non zero pixels in left belt.
    nonZero_l = cv.countNonZero(area_left)
    per_l = int((nonZero_l/area_l) * 100)

    # Counting Non zero pixel in right belt.
    nonZero_r = cv.countNonZero(area_right)
    per_r = int((nonZero_r/area_r) * 100)

    # return percentage of white pixels in left and right belt area.
    return per_l, per_r        
    
    

    



    

    

    
