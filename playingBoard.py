import os
import cv2
import numpy as np
import imutils
from copy import deepcopy
import time

BKG_THRESH = 60

class PlayingBoard:
    """
    This class represent the Playing Board. Saves the relevant values and parameters
    """
    def __init__(self):
        self.name = "board0"
        self.orig_board = []                          # This will save the original board that we will work with
        self.transformed_board = []               # This will save the transformed board that we will work with
        self.board_with_contour = []
        self.perspective_transform_matrix = []
        self.top_left_corner = []
        self.top_right_corner = []
        self.bottom_left_corner = []
        self.bottom_right_corner = []
        self.width = 0
        self.height = 0

    def set_name(self, name):
        self.name = name

    def set_orig_board(self, board):
        self.orig_board = board

    def set_transformed_board(self, trans_board):
        self.transformed_board = trans_board

    def set_board_with_contour(self, contour):
        self.board_with_contour = contour

    def set_width(self, width):
        self.width = width

    def set_height(self, height):
        self.height = height

    def set_perspective_transform_matrix(self, matrix):
        self.perspective_transform_matrix = matrix

def board_detection(frame):
    """This function will detect the playing board from the camera"""
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the grayscale image to reduce noise
    blurred = cv2.bilateralFilter(gray, 11, 17, 17)
    blurred = cv2.medianBlur(blurred, 9)
    blurred = cv2.GaussianBlur(blurred, (5, 5), 0)

    # Perform adaptive thresholding
    # Find the most common pixel value in the image to use as the background level

    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 2)

    # Invert the thresholded image
    thresh = cv2.bitwise_not(thresh)


    # Apply morphological operations to connect components
    kernel = np.ones((15, 15), np.uint8)
    connected = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(connected, connectivity=4)

    # Filter out small components (optional)
    min_size = 40000  # minimum size of connected component to consider
    filtered_labels = np.zeros_like(labels, dtype=np.uint8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            filtered_labels[labels == i] = 255


    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(filtered_labels, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Sort the contours by area in descending order
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:50]

    # Find the closed contour with the largest length
    max_length = 0
    most_significant_contour = None
    epsilon = 0.005
    board_contour = None

    four_edge_cont = []
    for contour in contours:
        length = cv2.arcLength(contour, True)
        curr_cnt = cv2.approxPolyDP(contour, epsilon=(epsilon*length), closed=True)
        # We want to make sure it has 4 corners to increase the chance of looking at a rectangle
        if length > max_length and len(curr_cnt) == 4:
            max_length = length
            most_significant_contour = contour
            board_contour = curr_cnt
            four_edge_cont.append(curr_cnt)
            break

    # Draw the most significant contour on the original frame
    marked_frame = frame.copy()
    if board_contour is not None:
        cv2.drawContours(marked_frame, [most_significant_contour], -1, (0, 255, 0), 2)
        cv2.imshow("Searching Board...", marked_frame)
        contour_points = board_contour.reshape(4, 2)
    else:
        cv2.imshow("Searching Board...", marked_frame)
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

    # Define the width and height of the output image
    width = 2400
    height = 1600

    transformed_board, persp_matrix = board_transformation(corners, frame, height, width)

    # Compute the area of the detected board contour
    contour_area = cv2.contourArea(board_contour)

    # Compute the total area of the image
    total_area = frame.shape[0] * frame.shape[1]
    cut_off_area_ratio = 0.3
    # Check if the detected board occupies more than 30% of the total image area
    if contour_area > cut_off_area_ratio * total_area:
        # Create a new PlayingBoard instance
        playing_board = PlayingBoard()
        # Setting the board name
        playing_board.set_name('Main Board')
        # Setting the original board
        playing_board.set_transformed_board(transformed_board)
        # Setting the board with contours marked on it
        playing_board.set_board_with_contour(marked_frame)
        # Setting height and width
        playing_board.set_width(width)
        playing_board.set_height(height)
        # Setting the Perspective Transform Matrix
        playing_board.set_perspective_transform_matrix(persp_matrix)

        return playing_board
    else:
        return None


def board_transformation(corners, frame, height, width):
    top_left = corners[0]
    top_right = corners[1]
    bottom_left = corners[2]
    bottom_right = corners[3]


    # Define the new corners of the rectangle in the output image
    new_corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)

    # Create a matrix to perform perspective transformation
    matrix = cv2.getPerspectiveTransform(np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32),
                                         new_corners)

    # Apply perspective transformation to the original frame
    output_frame = cv2.warpPerspective(frame, matrix, (width, height))

    return output_frame, matrix


def get_board(cap, time_window=14, detection_threshold=0.4):
    playing_board = None
    temp_playing_board = None
    start_time = time.time()
    detection_count = 0
    frame_count = 0

    while time.time() - start_time < time_window:
        # Reading the frame from the camera
        ret, frame = cap.read()
        #cv2.imshow('Name', frame)

        # Trying to get the playing board:
        temp_playing_board = board_detection(frame)

        if temp_playing_board:
            detection_count += 1
            print("Frame count: ", frame_count)
            print("Detection count: ", detection_count)
            playing_board = deepcopy(temp_playing_board)
            #cv2.imshow("Searching for Boards...", playing_board.board_with_contour)
            if (detection_count > 5) and (detection_count / frame_count >= detection_threshold):
                break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1
    # Release the webcam and close all OpenCV windows
    cv2.destroyAllWindows()

    if detection_count / frame_count >= detection_threshold:
        # Configure and display the contoured and transformed images if the playing surface was found
        cnt_disp = deepcopy(playing_board.board_with_contour)
        trans_disp = deepcopy(playing_board.transformed_board)
        display(cnt_disp, trans_disp)
        valid_surface = playing_board
        cv2.destroyAllWindows()
        return valid_surface
    else:
        print("Board wasn't found in at least 40% of the frames within the time window")
        return None

def display(contoured, transformed=np.array([])):
    # Arbitrary x, y offsets for displays
    x_offset = 50
    y_offset = 50

    # Original image with the contour of the playing surface overlayed
    #cv2.imshow('Contoured', contoured)
    # cv2.moveWindow("Original", x_offset, y_offset)

    # Only show the transformed surface if it exists (given to function implies existence)
    if bool(transformed.any()):
        # Transformed playing surface
        #cv2.imshow("Transformed", transformed)
        pass
    return