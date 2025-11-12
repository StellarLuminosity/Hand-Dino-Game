import numpy as np
import cv2

import math
import pyautogui

# Gesture‑to‑Action mapping: palm → jump, fist → duck, peace → neutral
# We classify based purely on convexity defects count:
#   count_defects >= 4 → palm (jump)
#   count_defects <= 1 → fist (duck)
#   otherwise → peace (neutral)

# Open Camera
capture = cv2.VideoCapture(0)

while capture.isOpened():

    # Capture frames from the camera
    ret, frame = capture.read()

    # Get hand data from the rectangle sub window
    cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 0)
    crop_image = frame[100:300, 100:300]

    # Apply Gaussian blur
    blur = cv2.GaussianBlur(crop_image, (3, 3), 0)

    # Change color-space from BGR -> HSV
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # Create a binary image with where white will be skin colors and rest is black
    mask2 = cv2.inRange(hsv, np.array([2, 0, 0]), np.array([20, 255, 255]))

    # Kernel for morphological transformation
    kernel = np.ones((5, 5))

    # Apply morphological transformations to filter out the background noise
    dilation = cv2.dilate(mask2, kernel, iterations=1)
    erosion = cv2.erode(dilation, kernel, iterations=1)

    # Apply Gaussian Blur and Threshold
    filtered = cv2.GaussianBlur(erosion, (3, 3), 0)
    ret, thresh = cv2.threshold(filtered, 127, 255, 0)
#####
    # Find contours
    #image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    try:
        # Find contour with maximum area
        contour = max(contours, key=lambda x: cv2.contourArea(x))

        # Create bounding rectangle around the contour
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(crop_image, (x, y), (x + w, y + h), (0, 0, 255), 0)

        # Find convex hull
        hull = cv2.convexHull(contour)

        # Draw contour
        drawing = np.zeros(crop_image.shape, np.uint8)
        cv2.drawContours(drawing, [contour], -1, (0, 255, 0), 0)
        cv2.drawContours(drawing, [hull], -1, (0, 0, 255), 0)

        # Fi convexity defects
        hull = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull)

        # Use cosine rule to find angle of the far point from the start and end point i.e. the convex points (the finger
        # tips) for all defects
        count_defects = 0

        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])

            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            angle = (math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180) / 3.14

            # if angle >= 90 draw a circle at the far point
            if angle <= 90:
                count_defects += 1
                cv2.circle(crop_image, far, 1, [0, 0, 255], -1)

            cv2.line(crop_image, start, end, [0, 255, 0], 2)

        # Press SPACE if condition is match

        # Decide gesture → action mapping
        action = None
        if count_defects >= 4:
            gesture = "palm"         # open hand
            action   = "jump"
        elif count_defects <= 1:
            gesture = "fist"         # closed hand
            action   = "duck"
        else:
            gesture = "peace"        # anything else
            action   = "neutral"

        # Print/log for debugging
        print(f"Detected gesture={gesture}, defects={count_defects}, action={action}")

        # Trigger game keystroke
        if action == "jump":
            pyautogui.press('space')
            cv2.putText(frame, "JUMP", (115, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2)
        elif action == "duck":
            # hold down arrow key briefly for duck
            pyautogui.keyDown('down')
            cv2.putText(frame, "DUCK", (115, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2)
            # you might add a small sleep to press down then release:
            # time.sleep(0.1)
            pyautogui.keyUp('down')
        else:
            # neutral: no keypress
            cv2.putText(frame, "NEUTRAL", (115, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2)

        #PLAY RACING GAMES (WASD)
        """
        if count_defects == 1:
            pyautogui.press('w')
            cv2.putText(frame, "W", (115, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, 2, 2)
        if count_defects == 2:
            pyautogui.press('s')
            cv2.putText(frame, "S", (115, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, 2, 2)
        if count_defects == 3:
            pyautogui.press('aw')
            cv2.putText(frame, "aw", (115, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, 2, 2)
        if count_defects == 4:
            pyautogui.press('dw')
            cv2.putText(frame, "dw", (115, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, 2, 2)
        if count_defects == 5:
            pyautogui.press('s')
            cv2.putText(frame, "s", (115, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, 2, 2)
        """

    except:
        pass

    # Show required images
    cv2.imshow("Gesture", frame)

    # Close the camera if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()