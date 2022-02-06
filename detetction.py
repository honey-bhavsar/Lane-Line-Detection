# Snippet to perform imports
import numpy as np
import cv2
import imutils
import time

# captured video of the road
cap = cv2.VideoCapture('testvideo2.mp4')

# Loop through the entire video
while (cap.isOpened()):
    # One by one read video frame
    ret, frame = cap.read()
    # cv2.imshow("Original Scene", frame)

    # Snip the segment of video frame of interest and display it on screen
    snip = frame[500:700, 300:900]
    cv2.imshow("Snip", snip)

    # To choose a region of interest, create a mask.
    mask = np.zeros((snip.shape[0], snip.shape[1]), dtype="uint8")
    pts = np.array([[25, 190], [275, 50], [380, 50], [575, 190]], dtype=np.int32)
    cv2.fillConvexPoly(mask, pts, 255)
    cv2.imshow("Mask", mask)

    # Apply a mask and display the masked image on the screen.
    masked = cv2.bitwise_and(snip, snip, mask=mask)
    cv2.imshow("Region of Interest", masked)

    # Convert the image to grayscale, then to black/white, and finally to binary.
    frame = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    thresh = 200
    frame = cv2.threshold(frame, thresh, 255, cv2.THRESH_BINARY)[1]
    cv2.imshow("Black/White", frame)

    # To aid with edge recognition, blur the image.
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    # cv2.imshow("Blurred", blurred)

    # Identify the edges and display them on the screen.
    edged = cv2.Canny(blurred, 30, 150)
    cv2.imshow("Edged", edged)

    # To find lane lines, perform the full Hough Transform.
    lines = cv2.HoughLines(edged, 1, np.pi / 180, 25)

    # Creating separate arrays for the left and right lanes.
    rho_left = []
    theta_left = []
    rho_right = []
    theta_right = []

    # to ensure that at least one line should have been detected by cv2.HoughLines.

    if lines is not None:
        # Loop through all of the lines found by cv2.HoughLines
        for i in range(0, len(lines)):
            # Evaluate each row of cv2.HoughLines output 'lines'
            for rho, theta in lines[i]:
                # For left lanes
                if theta < np.pi / 2 and theta > np.pi / 4:
                    rho_left.append(rho)
                    theta_left.append(theta)

                # For right lanes
                if theta > np.pi / 2 and theta < 3 * np.pi / 4:
                    rho_right.append(rho)
                    theta_right.append(theta)

    # In order to determine the median lane dimensions, statistics are used.

    left_rho = np.median(rho_left)
    left_theta = np.median(theta_left)
    right_rho = np.median(rho_right)
    right_theta = np.median(theta_right)

    # On top of the scene snip, plot a median lane.

    if left_theta > np.pi / 4:
        a = np.cos(left_theta);
        b = np.sin(left_theta)
        x0 = a * left_rho;
        y0 = b * left_rho
        offset1 = 250;
        offset2 = 800
        x1 = int(x0 - offset1 * (-b));
        y1 = int(y0 - offset1 * (a))
        x2 = int(x0 + offset2 * (-b));
        y2 = int(y0 + offset2 * (a))

        cv2.line(snip, (x1, y1), (x2, y2), (0, 255, 0), 6)

    if right_theta > np.pi / 4:
        a = np.cos(right_theta);
        b = np.sin(right_theta)
        x0 = a * right_rho;
        y0 = b * right_rho
        offset1 = 290;
        offset2 = 800
        x3 = int(x0 - offset1 * (-b));
        y3 = int(y0 - offset1 * (a))
        x4 = int(x0 - offset2 * (-b));
        y4 = int(y0 - offset2 * (a))

        cv2.line(snip, (x3, y3), (x4, y4), (255, 0, 0), 6)

    # Replacing a semi-transparent lane outline for the original.

    if left_theta > np.pi / 4 and right_theta > np.pi / 4:
        pts = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.int32)

        overlay = snip.copy()  # To create a copy of the original

        cv2.fillConvexPoly(overlay, pts, (0, 255, 0))  # Drawing shapes

        opacity = 0.4  # Blending with the original
        cv2.addWeighted(overlay, opacity, snip, 1 - opacity, 0, snip)

    cv2.imshow("Lined", snip)

    # To exit video mode, press the q key.

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()