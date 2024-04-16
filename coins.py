
import cv2
import numpy as np

class Coin:
    def __init__(self, radius, center, rank):
        self.radius = radius
        self.center = center
        self.rank = rank            # Saves the color of the coin, called rank to use the same terminology as the cards
        self.group = None

    def set_group(self, group):
        self.group = group


def group_coins(coins, image):
    dealer_coins = []
    player1_coins = []
    player2_coins = []
    board_height, board_width = image.shape[:2]

    upper_third_height = board_height / 3

    for coin in coins:
        if not isinstance(coin, Coin):
            continue
        coin_center_y = coin.center[1]
        coin_center_x = coin.center[0]

        # Check if the card is in the upper third of the image
        if coin_center_y < upper_third_height:
            dealer_coins.append(coin.rank)
        else:
            # Check if the card is on the left or right side of the image
            if coin_center_x < board_width / 2:
                player1_coins.append(coin.rank)
            else:
                player2_coins.append(coin.rank)

    return dealer_coins, [player1_coins, player2_coins]

def detect_coins(image):
    MIN_COIN_AREA = 1000
    MAX_COIN_AREA = 10000
    coins = []

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect circles using Hough Circle Transform with adjusted parameters
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                               param1=50, param2=40, minRadius=45, maxRadius=60)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            # Draw the circle and its center
            if y > 800:
                cv2.circle(image, (x, y), r, (0, 0, 255), 4)
                cv2.circle(image, (x, y), 2, (0, 0, 255), 3)

                # Extract ROI for the circle
                roi = gray[y - r:y + r, x - r:x + r]

                # Calculate average color in the ROI
                avg_color = np.mean(roi)

                # Determine the color based on the average color values
                if avg_color > 100:
                    color = "Light"
                else:
                    color = "Dark"

                coins.append(Coin(r, (x, y), color))

    dealer_coins, players_coins = group_coins(coins, image)

    # Display the result
    # cv2.imshow("Detected Circles", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return coins, dealer_coins, players_coins, image
