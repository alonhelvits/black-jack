import copy
import os
import cv2
import numpy as np
import playingBoard
import cards as cards_file
import coins as coins_file
import player
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim


def compare_frames(frame1, frame2):
    # Convert frames to grayscale
    gray_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Calculate Structural Similarity Index (SSIM)
    ssim_score = ssim(gray_frame1, gray_frame2)

    return ssim_score

def read_from_video(video_path):
    # Path to the recorded video file
    # video_path = 'train_files/one_round_game.MOV'

    # Initialize the video capture object
    cap = cv2.VideoCapture(video_path)
    # Check if the capture device is opened successfully
    if not cap.isOpened():
        print("Error: Failed to open capture device.")
        return None

    playing_board = playingBoard.get_board(cap)

    if playing_board is not None:
        print("Found the board!")
        print("Starting game.")
        print("Press 'q' to quit the game.")
        print("Press 'r' to reshuffle the deck.")
        print("Press 'b' to reset the board detection.")
        print("Press 'p' to reset the players total profit.")

        running_count = 0
        true_count = 0
        game_state_manager = player.GameState()
        decks_remaining = 2
        players_total_profit = [0, 0]
        previous_dealer_cards = []
        previous_players_cards = [[], []]
        previous_all_cards = []
        previous_dealer_coins = []
        previous_players_coins = [[], []]
        previous_all_coins = []
        prev_cards_contours = []
        prev_coins = []
        prev_frame = []
        failed_cap_detection_cnt = 0
        while True:
            if cap.isOpened():
                # Capture next frames
                ret, frame = cap.read()
                # Check if the frame is successfully captured
                if not ret:
                    print("Error: Failed to capture frame")
                    if failed_cap_detection_cnt > 10:
                        break
                    else:
                        failed_cap_detection_cnt += 1
                        continue
                    # cv2.imshow("Transformed", frame)
                failed_cap_detection_cnt = 0
                transformed_board = cv2.warpPerspective(frame, playing_board.perspective_transform_matrix,
                                                        (playing_board.width, playing_board.height))
                # plt.figure(figsize=(10, 10))
                # plt.imshow(cv2.cvtColor(transformed_board, cv2.COLOR_BGR2RGB))
                # plt.show()
                if prev_frame != []:
                    ssim = compare_frames(transformed_board, prev_frame)
                    if ssim > 0.96:
                        # Draw contours of cards and coins from previous frame
                        cv2.drawContours(transformed_board, prev_cards_contours, -1, (0, 0, 255), 3)
                        for coin in prev_coins:
                            (x, y), r = coin.center, coin.radius
                            if coin.rank == "Blue":
                                cv2.circle(transformed_board, (x, y), r, (255, 0, 0), 4)
                                cv2.circle(transformed_board, (x, y), 2, (255, 0, 0), 3)
                            elif coin.rank == "Green":
                                cv2.circle(transformed_board, (x, y), r, (0, 255, 0), 4)
                                cv2.circle(transformed_board, (x, y), 2, (0, 255, 0), 3)
                            else:
                                cv2.circle(transformed_board, (x, y), r, (0, 0, 255), 4)
                                cv2.circle(transformed_board, (x, y), 2, (0, 0, 255), 3)
                    else:
                        # apply card detection
                        coins, dealer_coins, players_coins, marked_coins_board = coins_file.detect_coins(
                            transformed_board)

                        # apply card detection
                        cards, dealer_cards, players_cards, marked_cards_board = cards_file.detect_cards(
                            transformed_board)
                else:
                    # apply card detection
                    coins, dealer_coins, players_coins, marked_coins_board = coins_file.detect_coins(transformed_board)

                    # apply card detection
                    cards, dealer_cards, players_cards, marked_cards_board = cards_file.detect_cards(transformed_board)

                # plt.figure(figsize=(10, 10))
                # plt.imshow(cv2.cvtColor(marked_cards_board, cv2.COLOR_BGR2RGB))
                # plt.show()

                # apply the game logic
                all_cards = dealer_cards + players_cards[0] + players_cards[1]
                # no situation of fewer cards than previous should happen in the middle of the game, card is not detected
                if len(previous_all_cards) > len(all_cards) > 0 and (
                        game_state_manager.is_playing() or game_state_manager.is_result()):
                    game_image, running_count, true_count, game_state_manager, decks_remaining, players_total_profit = player.process_game(
                        previous_dealer_cards, previous_players_cards, previous_dealer_coins, previous_players_coins,
                        marked_cards_board, running_count, true_count,
                        game_state_manager,
                        decks_remaining, players_total_profit, previous_players_cards, previous_dealer_cards)
                else:
                    game_image, running_count, true_count, game_state_manager, decks_remaining, players_total_profit = player.process_game(
                        dealer_cards, players_cards, dealer_coins, players_coins, marked_cards_board, running_count,
                        true_count, game_state_manager,
                        decks_remaining, players_total_profit, previous_players_cards, previous_dealer_cards)
                    # update previous cards, when normal game is running
                    previous_dealer_cards = dealer_cards
                    previous_players_cards = players_cards
                    previous_all_cards = all_cards

                new_width, new_height = 4800, 3200
                resized_image = cv2.resize(game_image, (new_width, new_height))
                cv2.imshow("Detected Cards", resized_image)

                # plt.figure(figsize=(10, 10))
                # plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
                # plt.show()

                prev_cards_contours = [card.contour for card in cards]
                prev_coins = coins
                prev_frame = copy.deepcopy(transformed_board)

                # Use cv2.waitKey() to check for keyboard input
                key = cv2.waitKey(1) & 0xFF

                # if the 'q' key is pressed, break from the loop
                if key == ord('q'):
                    break
                # if the 'i' key is pressed, perform an action
                elif key == ord('p'):
                    print("Resetting players total profit")
                    players_total_profit = [0, 0]
                elif key == ord('r'):
                    running_count = 0
                    true_count = 0
                    print("Deck Reshuffled, Running Count: 0, True Count: 0")
                elif key == ord('b'):
                    print("Resetting the board detection")
                    cap.release()
                    cv2.destroyAllWindows()
                    read_video_from_iphone()
        else:
            print("Recapturing the video")
            cap = cv2.VideoCapture(1)
    else:
        print("Fuck my life")
    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


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
        print("Press 'p' to reset the players total profit.")

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
        prev_cards_contours = []
        prev_coins = []
        prev_frame = []
        failed_cap_detection_cnt = 0
        while True:
            if cap.isOpened():
                # Capture next frames
                ret, frame = cap.read()
                # Check if the frame is successfully captured
                if not ret:
                    print("Error: Failed to capture frame")
                    if failed_cap_detection_cnt > 10:
                        break
                    else:
                        failed_cap_detection_cnt += 1
                        continue
                failed_cap_detection_cnt = 0
                transformed_board = cv2.warpPerspective(frame, playing_board.perspective_transform_matrix,
                                                        (playing_board.width, playing_board.height))

                if prev_frame != []:
                    ssim = compare_frames(transformed_board, prev_frame)
                    if ssim > 0.96:
                        # Draw contours of cards and coins from previous frame
                        cv2.drawContours(transformed_board, prev_cards_contours, -1, (0, 0, 255), 3)
                        for coin in prev_coins:
                            (x, y), r = coin.center, coin.radius
                            if coin.rank == "Blue":
                                cv2.circle(transformed_board, (x, y), r, (255, 0, 0), 4)
                                cv2.circle(transformed_board, (x, y), 2, (255, 0, 0), 3)
                            elif coin.rank == "Green":
                                cv2.circle(transformed_board, (x, y), r, (0, 255, 0), 4)
                                cv2.circle(transformed_board, (x, y), 2, (0, 255, 0), 3)
                            else:
                                cv2.circle(transformed_board, (x, y), r, (0, 0, 255), 4)
                                cv2.circle(transformed_board, (x, y), 2, (0, 0, 255), 3)
                    else:
                        # apply card detection
                        coins, dealer_coins, players_coins, marked_coins_board = coins_file.detect_coins(transformed_board)

                        # apply card detection
                        cards, dealer_cards, players_cards, marked_cards_board = cards_file.detect_cards(transformed_board)
                else:
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
                        decks_remaining, players_total_profit, previous_players_cards, previous_dealer_cards)
                else:
                    game_image, running_count, true_count, game_state_manager, decks_remaining, players_total_profit = player.process_game(
                        dealer_cards, players_cards, dealer_coins, players_coins, marked_cards_board, running_count,
                        true_count, game_state_manager,
                        decks_remaining, players_total_profit, previous_players_cards, previous_dealer_cards)
                    # update previous cards, when normal game is running
                    previous_dealer_cards = dealer_cards
                    previous_players_cards = players_cards
                    previous_all_cards = all_cards

                new_width, new_height = 4800, 3200
                resized_image = cv2.resize(game_image, (new_width, new_height))
                cv2.imshow("Detected Cards", resized_image)

                prev_cards_contours = [card.contour for card in cards]
                prev_coins = coins
                prev_frame = copy.deepcopy(transformed_board)

                # Use cv2.waitKey() to check for keyboard input
                key = cv2.waitKey(1) & 0xFF

                # if the 'q' key is pressed, break from the loop
                if key == ord('q'):
                    break
                # if the 'i' key is pressed, perform an action
                elif key == ord('p'):
                    print("Resetting players total profit")
                    players_total_profit = [0, 0]
                elif key == ord('r'):
                    running_count = 0
                    true_count = 0
                    print("Deck Reshuffled, Running Count: 0, True Count: 0")
                elif key == ord('b'):
                    print("Resetting the board detection")
                    cap.release()
                    cv2.destroyAllWindows()
                    read_video_from_iphone()
        else:
            print("Recapturing the video")
            cap = cv2.VideoCapture(1)
    else:
        print("Board wasn't found, Please make sure the camera is stable and the board is visible.")
    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


def playBlackJack():
    """
    This function reads the video from the webcam cameras and starts the game
    """

    read_video_from_iphone()


if __name__ == '__main__':
    playBlackJack()