import numpy as np
import cv2
import coins

test_image = cv2.imread("train_files/chips_image.png")
a = coins.find_coins(test_image)