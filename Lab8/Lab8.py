import cv2
import numpy as np
from matplotlib import pyplot as plt

# Step 4: Load the ATU image
img = cv2.imread('ATU.jpg')

# Step 5: Convert the ATU image to grayscale
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Step 6: Plotting ATU images with Matplotlib
plt.subplot(2, 2, 1), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('ATU Original'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 2), plt.imshow(imgGray, cmap='gray')
plt.title('ATU GrayScale'), plt.xticks([]), plt.yticks([])

# Step 7: Apply GaussianBlur with different kernel sizes to ATU image
kernel_sizes = [3, 9]

for i, kernel_size in enumerate(kernel_sizes):
    imgOut = cv2.GaussianBlur(imgGray, (kernel_size, kernel_size), 0)
    plt.subplot(2, 2, i + 3), plt.imshow(imgOut, cmap='gray')
    plt.title(f'ATU Blurred {kernel_size}x{kernel_size}'), plt.xticks([]), plt.yticks([])

plt.show()

# Step 8: Perform Sobel operator on the ATU image
sobelHorizontal = cv2.Sobel(imgGray, cv2.CV_64F, 1, 0, ksize=5)
sobelVertical = cv2.Sobel(imgGray, cv2.CV_64F, 0, 1, ksize=5)

# Step 9: Plot Sobel outputs for ATU
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1), plt.imshow(sobelHorizontal, cmap='gray')
plt.title('ATU Sobel Horizontal'), plt.xticks([]), plt.yticks([])

plt.subplot(1, 3, 2), plt.imshow(sobelVertical, cmap='gray')
plt.title('ATU Sobel Vertical'), plt.xticks([]), plt.yticks([])

# Step 10: Combine horizontal and vertical edges for ATU
sobelCombined = cv2.addWeighted(np.abs(sobelHorizontal), 0.5, np.abs(sobelVertical), 0.5, 0)
plt.subplot(1, 3, 3), plt.imshow(sobelCombined, cmap='gray')
plt.title('ATU Combined Sobel'), plt.xticks([]), plt.yticks([])

plt.show()

# Step 11: Perform Canny edge detection on the ATU image
cannyThreshold = 100
cannyParam2 = 200
canny = cv2.Canny(imgGray, cannyThreshold, cannyParam2)

# Step 12: Plot the Canny edge detection result for ATU
plt.imshow(canny, cmap='gray')
plt.title('ATU Canny Edge Detection'), plt.xticks([]), plt.yticks([])
plt.show()

# Step 13: Trial all the above with another image of your choice
# Step 8: Load the Galway image
img_galway = cv2.imread('galway.jpg')

# Step 9: Convert the Galway image to grayscale
imgGray_galway = cv2.cvtColor(img_galway, cv2.COLOR_BGR2GRAY)

# Step 10: Plotting Galway images with Matplotlib
plt.subplot(2, 2, 1), plt.imshow(cv2.cvtColor(img_galway, cv2.COLOR_BGR2RGB))
plt.title('Galway Original'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 2), plt.imshow(imgGray_galway, cmap='gray')
plt.title('Galway GrayScale'), plt.xticks([]), plt.yticks([])

# Step 11: Apply GaussianBlur with different kernel sizes to Galway image
kernel_sizes_galway = [3, 9]

for i, kernel_size in enumerate(kernel_sizes_galway):
    imgOut_galway = cv2.GaussianBlur(imgGray_galway, (kernel_size, kernel_size), 0)
    plt.subplot(2, 2, i + 3), plt.imshow(imgOut_galway, cmap='gray')
    plt.title(f'Galway Blurred {kernel_size}x{kernel_size}'), plt.xticks([]), plt.yticks([])

plt.show()

# Step 12: Perform Sobel operator on the Galway image
sobelHorizontal_galway = cv2.Sobel(imgGray_galway, cv2.CV_64F, 1, 0, ksize=5)
sobelVertical_galway = cv2.Sobel(imgGray_galway, cv2.CV_64F, 0, 1, ksize=5)

# Step 13: Plot Sobel outputs for Galway
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1), plt.imshow(sobelHorizontal_galway, cmap='gray')
plt.title('Galway Sobel Horizontal'), plt.xticks([]), plt.yticks([])

plt.subplot(1, 3, 2), plt.imshow(sobelVertical_galway, cmap='gray')
plt.title('Galway Sobel Vertical'), plt.xticks([]), plt.yticks([])

# Step 14: Combine horizontal and vertical edges for Galway
sobelCombined_galway = cv2.addWeighted(np.abs(sobelHorizontal_galway), 0.5, np.abs(sobelVertical_galway), 0.5, 0)
plt.subplot(1, 3, 3), plt.imshow(sobelCombined_galway, cmap='gray')
plt.title('Galway Combined Sobel'), plt.xticks([]), plt.yticks([])

plt.show()

# Step 15: Perform Canny edge detection on the Galway image
cannyThreshold_galway = 100
cannyParam2_galway = 200
canny_galway = cv2.Canny(imgGray_galway, cannyThreshold_galway, cannyParam2_galway)

# Step 16: Plot the Canny edge detection result for Galway
plt.imshow(canny_galway, cmap='gray')
plt.title('Galway Canny Edge Detection'), plt.xticks([]), plt.yticks([])
plt.show()

# Threshold the Sobel sum image and visualize for different thresholds
threshold_values = [10, 50, 100, 200, 300]

for threshold in threshold_values:
    sobel_sum_thresholded = np.where(sobelCombined > threshold, 1, 0)
    plt.imshow(sobel_sum_thresholded, cmap='gray')
    plt.title(f'ATU Sobel Sum Thresholded (Threshold={threshold})')
    plt.show()