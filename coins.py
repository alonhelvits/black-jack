import cv2
import numpy as np
from cards import group_cards_coins

class Coins:
    def __init__(self, radius, center, rank):
        self.radius = radius
        self.center = center
        self.rank = rank            # Saves the color of the coin, called rank to use the same terminology as the cards
        self.group = None

    def set_group(self, group):
        self.group = group


def detect_coins(image):
    # TBD
    MIN_COIN_Area = 1000
    MAX_COIN_Area = 10000
    coins = []


    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Convert the image to grayscale
    # Apply Gaussian blur

    # Apply Gaussian blur to reduce noise
    #blur = cv2.GaussianBlur(gray, (5, 5), 0)
    blur = cv2.bilateralFilter(gray, 11, 17, 11)

    # Detect edges using Canny edge detector
    edges = cv2.Canny(blur, 60, 150)
    #cv2.imshow("Detected Circles", edges)
    #cv2.waitKey(0)


    # Apply Hough Circle Transform
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=40,
                               param1=50, param2=43, minRadius=10, maxRadius=35)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        rad = np.max(circles[:, -1]) + 5
        box_rad = rad - 10
        for (x, y, r) in circles:
            # Draw the circle
            cv2.circle(image, (x, y), rad, (0, 255, 0), 2)
            # Draw the center of the circle
            cv2.circle(image, (x, y), 2, (0, 0, 255), 3)
            #cv2.imshow("Detected Circles", image)
            #cv2.waitKey(0)
            # Extract ROI for the circle
            roi = image[y - box_rad:y + box_rad, x - box_rad:x + box_rad]

            # Calculate average color in the ROI
            avg_color = np.mean(roi, axis=(0, 1))

            # Determine the color based on the average color values
            if avg_color[0] > avg_color[1] and avg_color[0] > avg_color[2]:
                color = "Blue"
            elif avg_color[1] > avg_color[0] and avg_color[1] > avg_color[2]:
                color = "Green"
            else:
                color = "Red"

            coins.append(Coins(r, (x, y), color))
            #print("Coin color:", color)

    dealer_coins, player_coins = group_cards_coins(coins, image)

    # Display the result
    # cv2.imshow("Detected Circles", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return coins, dealer_coins, player_coins, image
