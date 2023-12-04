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

# Exercise 10: Create a deep copy for Shi Tomasi
imgShiTomasi = img.copy()

# Exercise 11: Plot Shi Tomasi corners
for i in corners:
    x, y = np.intp(i.ravel())  # Ensure integer values
    cv2.circle(imgShiTomasi, (x, y), 3, (0, 0, 255), -1)

# Exercise 12: Display Shi Tomasi corners
cv2.imshow('Shi Tomasi Corners', imgShiTomasi)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Exercise 13-14: ORB feature detection and plotting
orb = cv2.ORB_create()
keypoints, descriptors = orb.detectAndCompute(gray_img, None)
imgORB = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0), flags=0)

# Exercise 15: Display ORB keypoints
cv2.imshow('ORB Keypoints', imgORB)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Exercise 3: Load the image
img = cv2.imread('ATU2.jpg')

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

# Exercise 10: Create a deep copy for Shi Tomasi
imgShiTomasi = img.copy()

# Exercise 11: Plot Shi Tomasi corners
for i in corners:
    x, y = np.intp(i.ravel())  # Ensure integer values
    cv2.circle(imgShiTomasi, (x, y), 3, (0, 0, 255), -1)

# Exercise 12: Display Shi Tomasi corners
cv2.imshow('Shi Tomasi Corners', imgShiTomasi)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Exercise 13-14: ORB feature detection and plotting
orb = cv2.ORB_create()
keypoints, descriptors = orb.detectAndCompute(gray_img, None)
imgORB = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0), flags=0)

# Exercise 15: Display ORB keypoints
cv2.imshow('ORB Keypoints', imgORB)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Exercise 17: Feature matching with ORB
img1 = cv2.imread('ATU1.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('ATU2.jpg', cv2.IMREAD_GRAYSCALE)

# Use ORB to detect keypoints and descriptors
orb = cv2.ORB_create()
keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

# Use BruteForceMatcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)

# Sort them in ascending order of distance
matches = sorted(matches, key=lambda x: x.distance)

# Draw the matches
img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Display the matches
cv2.imshow('ORB Matches', img_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Advanced Exercise 1: Contour Detection
def thresh_callback(val):
    threshold = val
    # Detect edges using Canny
    canny_output = cv2.Canny(src_gray, threshold, threshold * 2)
    # Find contours
    contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Draw contours
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    for i in range(len(contours)):
        color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
        cv2.drawContours(drawing, contours, i, color, 2, cv2.LINE_8, hierarchy, 0)
    # Show in a window
    cv2.imshow('Contours', drawing)

# Load source image
src = cv2.imread('ATU1.jpg')  # Replace 'your_image.jpg' with the actual filename and path
if src is None:
    print('Could not open or find the image.')
    exit(0)

# Convert image to gray and blur it
src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
src_gray = cv2.blur(src_gray, (3, 3))

# Create Window
source_window = 'Source'
cv2.namedWindow(source_window)
cv2.imshow(source_window, src)

max_thresh = 255
thresh = 100  # initial threshold
cv2.createTrackbar('Canny Thresh:', source_window, thresh, max_thresh, thresh_callback)
thresh_callback(thresh)

cv2.waitKey(0)
cv2.destroyAllWindows()

#Do the same for the second image
def thresh_callback(val):
    threshold = val
    # Detect edges using Canny
    canny_output = cv2.Canny(src_gray, threshold, threshold * 2)
    # Find contours
    contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Draw contours
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    for i in range(len(contours)):
        color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
        cv2.drawContours(drawing, contours, i, color, 2, cv2.LINE_8, hierarchy, 0)
    # Show in a window
    cv2.imshow('Contours', drawing)

# Load source image
src = cv2.imread('ATU2.jpg')  # Replace 'your_image.jpg' with the actual filename and path
if src is None:
    print('Could not open or find the image.')
    exit(0)

# Convert image to gray and blur it
src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
src_gray = cv2.blur(src_gray, (3, 3))

# Create Window
source_window = 'Source'
cv2.namedWindow(source_window)
cv2.imshow(source_window, src)

max_thresh = 255
thresh = 100  # initial threshold
cv2.createTrackbar('Canny Thresh:', source_window, thresh, max_thresh, thresh_callback)
thresh_callback(thresh)

cv2.waitKey(0)
cv2.destroyAllWindows()

#Advanced Question part 2 image 1
# Load the image
img = cv2.imread('ATU1.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convert to HSV
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Split HSV channels
h, s, v = cv2.split(img_hsv)

# Display the result using matplotlib subplot
plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.imshow(img_rgb)
plt.title('Original Image')

plt.subplot(132)
plt.imshow(h, cmap='gray')
plt.title('H Channel')

plt.subplot(133)
plt.imshow(s, cmap='gray')
plt.title('S Channel')

plt.show()

#Advanced Question part 2 image 2
# Load the image
img = cv2.imread('ATU2.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convert to HSV
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Split HSV channels
h, s, v = cv2.split(img_hsv)

# Display the result using matplotlib subplot
plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.imshow(img_rgb)
plt.title('Original Image')

plt.subplot(132)
plt.imshow(h, cmap='gray')
plt.title('H Channel')

plt.subplot(133)
plt.imshow(s, cmap='gray')
plt.title('S Channel')

plt.show()