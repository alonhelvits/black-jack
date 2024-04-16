import os
import cv2
import numpy as np
import playingBoard
import cards as cards_file
import coins as coins_file
import player



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
        print("Found the board!")
        print("Starting game.")
        print("Press 'q' to quit the game.")
        print("Press 'r' to reshuffle the deck.")
        print("Press 'b' to reset the board detection.")

        running_count = 0
        skip_frames = 300
        true_count = 0
        game_state_manager = player.GameState()
        decks_remaining = 2
        players_total_profit = [0,0]
        previous_dealer_cards = []
        previous_players_cards = [[], []]
        previous_all_cards = []

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
                coins, dealer_coins, players_coins, marked_coins_board = coins_file.detect_coins(transformed_board)

                # apply card detection
                cards, dealer_cards, players_cards, marked_cards_board = cards_file.detect_cards(transformed_board)

                # apply the game logic
                all_cards = dealer_cards + players_cards[0] + players_cards[1]
                # no situation of fewer cards than previous should happen in the middle of the game, card is not detected
                if len(previous_all_cards) > len(all_cards) > 0 and (
                        game_state_manager.is_playing() or game_state_manager.is_result()):
                    game_image, running_count, true_count, game_state_manager, decks_remaining, players_total_profit = player.process_game(
                        previous_dealer_cards, previous_players_cards, dealer_coins, players_coins,
                        marked_cards_board, running_count, true_count,
                        game_state_manager,
                        decks_remaining, players_total_profit)
                else:
                    game_image, running_count, true_count, game_state_manager, decks_remaining, players_total_profit = player.process_game(
                        dealer_cards, players_cards, dealer_coins, players_coins, marked_cards_board, running_count,
                        true_count, game_state_manager,
                        decks_remaining, players_total_profit)
                    # update previous cards, when normal game is running
                    previous_dealer_cards = dealer_cards
                    previous_players_cards = players_cards
                    previous_dealer_coins = dealer_coins
                    previous_players_coins = players_coins
                    previous_all_cards = all_cards

                new_width, new_height = 1200, 800
                resized_image = cv2.resize(game_image, (new_width, new_height))
                cv2.imshow("Detected Cards", resized_image)

                # Use cv2.waitKey() to check for keyboard input
                key = cv2.waitKey(1) & 0xFF

                # if the 'q' key is pressed, break from the loop
                if key == ord('q'):
                    break
                # if the 'i' key is pressed, perform an action
                elif key == ord('r'):
                    running_count = 0
                    true_count = 0
                    players_total_profit = [0, 0]
                    decks_remaining = 2
                    print("Deck Reshuffled, Running Count: 0, True Count: 0")
                elif key == ord('b'):
                    print("Resetting the board detection")
                    cap.release()
                    cv2.destroyAllWindows()
                    read_from_video(video_path)
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

    print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Check if properties are set successfully
    if cap.get(cv2.CAP_PROP_FRAME_WIDTH) != 1920 or cap.get(cv2.CAP_PROP_FRAME_HEIGHT) != 1080:
        print("Error: Failed to set capture device properties for iPhone video.")
        cap.release()
        return None

    playing_board = playingBoard.get_board(cap)

    if playing_board is not None:
        print("Found the board!")
        print("Starting game.")
        print("Press 'q' to quit the game.")
        print("Press 'r' to reshuffle the deck.")
        print("Press 'b' to reset the board detection.")

        running_count = 0
        true_count = 0
        game_state_manager = player.GameState()
        decks_remaining = 2
        players_total_profit = [0,0]
        previous_dealer_cards = []
        previous_players_cards = [[], []]
        previous_all_cards = []
        previous_dealer_coins = []
        previous_players_coins = [[], []]
        previous_all_coins = []

        while True:
            # Capture next frames
            ret, frame = cap.read()
            # Check if the frame is successfully captured
            if not ret:
                print("Error: Failed to capture frame")
                break
                # cv2.imshow("Transformed", frame)

            transformed_board = cv2.warpPerspective(frame, playing_board.perspective_transform_matrix,
                                                    (playing_board.width, playing_board.height))

            # apply card detection
            coins, dealer_coins, players_coins, marked_coins_board = coins_file.detect_coins(transformed_board)

            # apply card detection
            cards, dealer_cards, players_cards, marked_cards_board = cards_file.detect_cards(transformed_board)

            # apply the game logic
            all_cards = dealer_cards + players_cards[0] + players_cards[1]
            # no situation of fewer cards than previous should happen in the middle of the game, card is not detected
            if len(previous_all_cards) > len(all_cards) > 0 and (
                    game_state_manager.is_playing() or game_state_manager.is_result()):
                game_image, running_count, true_count, game_state_manager, decks_remaining, players_total_profit = player.process_game(
                    previous_dealer_cards, previous_players_cards, previous_dealer_coins, previous_players_coins,
                    marked_cards_board, running_count, true_count,
                    game_state_manager,
                    decks_remaining, players_total_profit)
            else:
                game_image, running_count, true_count, game_state_manager, decks_remaining, players_total_profit = player.process_game(
                    dealer_cards, players_cards, dealer_coins, players_coins, marked_cards_board, running_count,
                    true_count, game_state_manager,
                    decks_remaining, players_total_profit)
                # update previous cards, when normal game is running
                previous_dealer_cards = dealer_cards
                previous_players_cards = players_cards
                previous_all_cards = all_cards

            new_width, new_height = 1200, 800
            resized_image = cv2.resize(game_image, (new_width, new_height))
            cv2.imshow("Detected Cards", resized_image)

            # Use cv2.waitKey() to check for keyboard input
            key = cv2.waitKey(1) & 0xFF

            # if the 'q' key is pressed, break from the loop
            if key == ord('q'):
                break
            # if the 'i' key is pressed, perform an action
            elif key == ord('r'):
                running_count = 0
                true_count = 0
                print("Deck Reshuffled, Running Count: 0, True Count: 0")
            elif key == ord('b'):
                print("Resetting the board detection")
                cap.release()
                cv2.destroyAllWindows()
                read_video_from_iphone()
        # Release the webcam and close all OpenCV windows
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Fuck my life")


def playBlackJack():
    """
    This function reads the video from the webcam cameras and starts the game
    """

    # read_and_write_video()
    #read_video_from_iphone()
    read_from_video('IMG_9172.MOV')
    # read_from_video('train_files/board_with_tape_one_round.MOV')


if __name__ == '__main__':
    playBlackJack()