import os
import cv2
import numpy as np


class PlayingBoard:
    """
    This class represent the Playing Board. Saves the relevant values and parameters
    """
    def __init__(self):
        self.orig_board = []                          # This will save the original board that we will work with
        self.transformed_board = []               # This will save the transformed board that we will work with
        self.top_left_corner = []
        self.top_right_corner = []
        self.bottom_left_corner = []
        self.bottom_right_corner = []


def board_detection(frame):
    """This function will detect the playing board from the camera"""
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform edge detection using Canny
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest length
    max_length = 0
    most_significant_contour = None
    epsilon = 0.02

    for contour in contours:
        length = cv2.arcLength(contour, True)
        curr_cnt = cv2.approxPolyDP(contour, epsilon=(epsilon*length), closed=True)
        if length > max_length and len(curr_cnt) == 4:
            max_length = length
            most_significant_contour = contour
            board_contour = curr_cnt


    # Draw the most significant contour on the original frame
    marked_frame = frame.copy()
    if board_contour is not None:
        cv2.drawContours(marked_frame, [most_significant_contour], -1, (0, 255, 0), 2)
    else:
        return None

    return marked_frame

