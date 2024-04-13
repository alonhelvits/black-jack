import numpy as np
import cv2
import coins

test_image = cv2.imread("train_files/chips.png")
a = coins.detect_coins(test_image)