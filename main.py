import os
import cv2
import numpy as np
import playingBoard
import cards as cards_file

def read_and_write_video():
    # Path to the recorded video file
    video_path = 'train_files/one_round_game.MOV'
    output_video_path = 'train_files/one_round_game_transformed.mov'  # Path to save the output video

    # Initialize the video capture object
    cap = cv2.VideoCapture(video_path)
    # Check if the video capture object is opened successfully
    if not cap.isOpened():
        print("Error: Failed to open video file")
        exit()

    '''
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 9999)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 9999)
    '''

    playing_board = playingBoard.get_board(cap)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Specify the codec (codec depends on the extension of the output file)
    out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (playing_board.width, playing_board.height))

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
            # Write the transformed frame to the output video
            out.write(transformed_board)

            cv2.imshow("Transformed", transformed_board)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # Release the webcam and close all OpenCV windows
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    else:
        print("Fuck my life")


def read_from_video():
    # Path to the recorded video file
    video_path = 'scattered_cards_transformed.MOV'

    # Initialize the video capture object
    cap = cv2.VideoCapture(video_path)
    # Check if the video capture object is opened successfully
    if not cap.isOpened():
        print("Error: Failed to open video file")
        exit()

    playing_board = playingBoard.get_board(cap)
    while True:
        # Capture next frames
        ret, frame = cap.read()
        # Check if the frame is successfully captured
        if not ret:
            print("Error: Failed to capture frame")
            break

        # cv2.imshow("Transformed", frame)

        # Transforming the new read image according to the transformation matrix
        #transformed_board = cv2.warpPerspective(frame, playing_board.perspective_transform_matrix,(playing_board.width, playing_board.height))
        cards, dealer_cards, players_cards, marked_cards_board = cards_file.Detect_cards(frame)
        cv2.imshow("marked_cards_board", marked_cards_board)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    '''
    if playing_board is not None:
        print("Found the board!!")
        while True:
            # Capture next frames
            ret, frame = cap.read()
            # Check if the frame is successfully captured
            if not ret:
                print("Error: Failed to capture frame")
                break

            # cv2.imshow("Transformed", frame)

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

def read_video_from_iphone():
    # Open video capture device (change index if necessary, 0 is usually the default webcam)
    cap = cv2.VideoCapture(0)

    # Check if the capture device is opened successfully
    if not cap.isOpened():
        print("Error: Failed to open capture device.")
        return None

    # Set capture device properties for iPhone video
    cap.set(cv2.CAP_PROP_FPS, 30)  # Set desired frame rate (adjust as needed)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Set desired frame width
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Set desired frame height

    # Check if properties are set successfully
    if cap.get(cv2.CAP_PROP_FRAME_WIDTH) != 1920 or cap.get(cv2.CAP_PROP_FRAME_HEIGHT) != 1080:
        print("Error: Failed to set capture device properties for iPhone video.")
        cap.release()
        return None

    playing_board = playingBoard.get_board(cap)

    if playing_board is not None:
        iter = 0
        print("Found the board!!")
        while True:
            #Capture next frames
            if iter == 10:
                iter = 0
            ret, frame = cap.read()
            # Check if the frame is successfully captured
            if not ret:
                print("Error: Failed to capture frame")
                break

            # Transforming the new read image according to the transformation matrix
            transformed_board = cv2.warpPerspective(frame, playing_board.perspective_transform_matrix,
                                                    (playing_board.width, playing_board.height))
            cards , dealer_cards, players_cards , marked_cards_board = cards_file.Detect_cards(transformed_board)
            cv2.imshow("marked_cards_board", marked_cards_board)
            iter += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # Release the webcam and close all OpenCV windows
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Fuck my life")


def playBlackJack():
    """
    This function reads the video from the webcam cameras and starts the game
    """
    #read_and_write_video()
    #read_video_from_iphone()
    read_from_video()


if __name__ == '__main__':
    playBlackJack()

