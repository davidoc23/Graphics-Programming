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