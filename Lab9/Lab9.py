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

# Exercise 6: Create a deep copy
imgHarris = img.copy()

# Exercise 7: Plot Harris corners
threshold = 0.01  # You can experiment with different values
for i in range(len(dst)):
    for j in range(len(dst[i])):
        if dst[i][j] > (threshold * dst.max()):
            cv2.circle(imgHarris, (j, i), 3, (0, 255, 0), -1)

# Exercise 8: Display Harris corners
cv2.imshow('Harris Corners', imgHarris)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Exercise 9: Shi Tomasi corner detection
maxCorners = 100
qualityLevel = 0.01
minDistance = 10
corners = cv2.goodFeaturesToTrack(gray_img, maxCorners, qualityLevel, minDistance)
