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