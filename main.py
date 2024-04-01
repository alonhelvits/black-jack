import os
import cv2
import numpy as np
import playingBoard


def playBlackJack():
    """
    This function reads the video from the webcam cameras and starts the game
    """
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 9999)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 9999)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Check if the frame is successfully captured
        if not ret:
            print("Error: Failed to capture frame")
            break

        # Detect contours in the frame
        marked_frame = playingBoard.board_detection(frame)

        # Display the frame
        cv2.imshow('Webcam', marked_frame)

        # Check for key press; if 'q' is pressed, exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    playBlackJack()

