import os
import cv2
import numpy as np
import playingBoard
import cards as cards_file
import player


def read_and_write_video():
    '''
    This function reads the video from a given file and writes the transformed video to a file
    :return:
    '''
    # Path to the recorded video file
    video_path = 'train_files/one_round_game.MOV'
    output_video_path = 'train_files/one_round_game_transformed.mov'  # Path to save the output video

    # Initialize the video capture object
    cap = cv2.VideoCapture(video_path)
    # Check if the video capture object is opened successfully
    if not cap.isOpened():
        print("Error: Failed to open video file")
        exit()

    playing_board = playingBoard.get_board(cap)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Specify the codec (codec depends on the extension of the output file)
    out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (playing_board.width, playing_board.height))

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
            # Write the transformed frame to the output video
            out.write(transformed_board)

            # cv2.imshow("Transformed", transformed_board)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # Release the webcam and close all OpenCV windows
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    else:
        print("Fuck my life")


def read_from_video(video_path):
    # Path to the recorded video file
    # video_path = 'train_files/one_round_game.MOV'

    # Initialize the video capture object
    cap = cv2.VideoCapture(video_path)
    # Check if the video capture object is opened successfully
    if not cap.isOpened():
        print("Error: Failed to open video file")
        exit()

    frame_count = 0
    playing_board = playingBoard.get_board(cap)

    if playing_board is not None:
        print("Found the board!!")

        running_count = 0
        skip_frames = 12
        true_count = 0
        game_state_manager = player.GameState()
        decks_remaining = 2

        while True:
            # Capture next frames
            ret, frame = cap.read()
            # Check if the frame is successfully captured
            if not ret:
                print("Error: Failed to capture frame")
                break

            # Process only every n-th frame
            if frame_count % skip_frames == 0:

                # cv2.imshow("Transformed", frame)

                transformed_board = cv2.warpPerspective(frame, playing_board.perspective_transform_matrix,
                                                        (playing_board.width, playing_board.height))

                # apply card detection
                cards, dealer_cards, players_cards, marked_cards_board = cards_file.Detect_cards(transformed_board)

                # apply the game logic
                game_image, running_count, true_count, game_state_manager, decks_remaining = player.process_game(
                    dealer_cards, players_cards, marked_cards_board, running_count, true_count, game_state_manager,
                    decks_remaining)
                new_width, new_height = 1200, 800

                resized_image = cv2.resize(game_image, (new_width, new_height))

                cv2.imshow("Detected Cards", resized_image)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            # Increment frame count
            frame_count += 1
        # Release the webcam and close all OpenCV windows
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Fuck my life")


def read_video_from_iphone():
    # Open video capture device (change index if necessary, 0 is usually the default webcam)
    cap = cv2.VideoCapture(1)

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
        print("Found the board!!")
        while True:
            # Capture next frames
            ret, frame = cap.read()
            # Check if the frame is successfully captured
            if not ret:
                print("Error: Failed to capture frame")
                break

            # Transforming the new read image according to the transformation matrix
            transformed_board = cv2.warpPerspective(frame, playing_board.perspective_transform_matrix,
                                                    (playing_board.width, playing_board.height))
            cards, dealer_cards, players_cards, marked_cards_board = cards_file.Detect_cards(transformed_board)
            new_width, new_height = 600, 400
            resized_image = cv2.resize(marked_cards_board, (new_width, new_height))
            cv2.imshow("Detected Cards", resized_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # Release the webcam and close all OpenCV windows
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Fuck my life")


def check_consecutive_matches(last_results):
    match_long = 3
    if len(last_results) > match_long:
        last_results.pop(0)  # Remove the oldest result

    # Check if the last match_long results are the same
    if len(last_results) == match_long and all(last_results[i] == last_results[0] for i in range(1, match_long)):
        is_consecutive_matches = True
    else:
        is_consecutive_matches = False
    return last_results, is_consecutive_matches


def playBlackJack():
    """
    This function reads the video from the webcam cameras and starts the game
    """

    # read_and_write_video()
    # read_video_from_iphone()
    # read_from_video('train_files/no_masking_tape.mov')
    read_from_video('clear_background_2.MOV')


if __name__ == '__main__':
    playBlackJack()
