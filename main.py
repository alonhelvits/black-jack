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

    playing_board = playingBoard.get_board(cap)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Check if the frame is successfully captured
        if not ret:
            print("Error: Failed to capture frame")
            break

        # Display the frame
        cv2.imshow('Webcam', frame)

        # Check for key press; if 'q' is pressed, exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

    '''
    if playing_board is not None:
        print("Found the board!!")
        while True:
            #Capture next frames
            ret, frame = cap.read()
            # Check if the frame is successfully captured
            if not ret:
                print("Error: Failed to capture frame")
                break

            #cv2.imshow("Transformed", frame)

            # Transforming the new read image according to the transformation matrix
            transformed_board = cv2.warpPerspective(frame, playing_board.perspective_transform_matrix,
                                                    (playing_board.width, playing_board.height))

            cv2.imshow("Transformed", transformed_board)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # Release the webcam and close all OpenCV windows
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Fuck my life")
    '''

if __name__ == '__main__':
    playBlackJack()

