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
    MIN_COIN_AREA = 1000
    MAX_COIN_AREA = 10000
    coins = []

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #Apply Gaussian blur to the grayscale image to reduce noise
    blurred = cv2.medianBlur(gray, 9)
    blurred = cv2.bilateralFilter(blurred, 11, 17, 17)

    #Detect edges using the Canny edge detector with tuned thresholds
    edges = cv2.Canny(blurred, 30, 150)

    #Apply morphological closing to enhance edge connectivity
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("Edges", edges)
    cv2.waitKey(0)

    # Detect circles using Hough Circle Transform with adjusted parameters
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                               param1=50, param2=40, minRadius=45, maxRadius=60)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            # Draw the circle and its center
            cv2.circle(image, (x, y), r, (0, 255, 0), 2)
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

            coins.append(Coins(r, (x, y), color))

    dealer_coins, player_coins = group_cards_coins(coins, image)

    # Display the result
    # cv2.imshow("Detected Circles", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return coins, dealer_coins, player_coins, image
