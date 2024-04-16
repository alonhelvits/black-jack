import numpy as np
import cv2
import coins
from main import read_from_video

# Initialize the video capture object
cap = cv2.VideoCapture('train_files/IMG_7149.mov')
# Check if the video capture object is opened successfully
if not cap.isOpened():
    print("Error: Failed to open video file")
    exit()

while True:
    ret, frame = cap.read()
    # Check if the frame is successfully captured
    if not ret:
        print("Error: Failed to capture frame")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the grayscale image to reduce noise
    blurred = cv2.bilateralFilter(gray, 11, 17, 17)
    blurred = cv2.medianBlur(blurred, 9)
    blurred = cv2.GaussianBlur(blurred, (5, 5), 0)

    #thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 2)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # cv2.imshow("Thresholded", thresh)
    # cv2.waitKey(0)

    # Detect circles using Hough Circle Transform with adjusted parameters
    circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                               param1=30, param2=20, minRadius=40, maxRadius=80)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            # Draw the circle and its center
            cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
            cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)

            # Extract ROI for the circle
            roi = gray[y - r:y + r, x - r:x + r]

    # Display the result
    cv2.imshow("Detected Circles", frame)
    # Use cv2.waitKey() to check for keyboard input
    key = cv2.waitKey(1) & 0xFF

    # if the 'q' key is pressed, break from the loop
    if key == ord('q'):
        break

cv2.destroyAllWindows()
# test_image = cv2.imread("train_files/ziton.png")
# a = coins.detect_coins(test_image)