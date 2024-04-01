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

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    # Find the closed contour with the largest length
    max_length = 0
    most_significant_contour = None
    epsilon = 0.02
    board_contour = []

    for contour in contours:
        length = cv2.arcLength(contour, True)
        curr_cnt = cv2.approxPolyDP(contour, epsilon=(epsilon*length), closed=True)
        # We want to make sure it has 4 corners to increase the chance of looking at a rectangle
        if length > max_length and len(curr_cnt) == 4:
            max_length = length
            most_significant_contour = contour
            board_contour = curr_cnt
            break


    # Draw the most significant contour on the original frame
    marked_frame = frame.copy()
    if board_contour is not None:
        cv2.drawContours(marked_frame, [most_significant_contour], -1, (0, 255, 0), 2)
        contour_points = board_contour.reshape(4, 2)
    else:
        return None

    # Finding the bottom left and top right corners
    sum_each_corner = np.sum(contour_points, axis=1)
    top_left_index = np.argmin(sum_each_corner)
    bottom_right_index = np.argmax(sum_each_corner)
    top_left_corner = contour_points[top_left_index]
    bottom_right_corner = contour_points[bottom_right_index]

    # Temporary variables for intermediate steps
    temp = None
    reduced_points = None

    # Removing top left corner from the array
    for i in range(contour_points.shape[0]):
        if np.array_equal(contour_points[i], top_left_corner):
            temp = np.delete(contour_points, i, 0)
            break

    # Removing bottom right corner from the modified array
    for i in range(temp.shape[0]):
        if np.array_equal(temp[i], bottom_right_corner):
            reduced_points = np.delete(temp, i, 0)
            break

    # Compute the difference between the points -- the top-right will have the minimum difference
    # and the bottom-left will have the maximum difference
    diff = np.diff(reduced_points, axis=1)
    top_right_corner = reduced_points[np.argmin(diff)]
    bottom_left_corner = reduced_points[np.argmax(diff)]

    corners = np.array([top_left_corner, top_right_corner, bottom_left_corner, bottom_right_corner])

    #testtt

    return marked_frame

