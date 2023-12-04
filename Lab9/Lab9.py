import cv2
import numpy as np
import matplotlib.pyplot as plt

# Exercise 3: Load the image
img = cv2.imread('ATU1.jpg')

# Exercise 4: Convert the image to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Exercise 5: Harris corner detection
blockSize = 2
aperture_size = 3
k = 0.04
dst = cv2.cornerHarris(gray_img, blockSize, aperture_size, k)
