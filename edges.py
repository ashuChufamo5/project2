import cv2
import numpy as np

# Load the image
image = cv2.imread('image.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Canny edge detection
edges = cv2.Canny(gray, 100, 200)

# Display the original and edge-detected images
cv2.imshow('Original Image', image)
cv2.imshow('Edge-Detected Image', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()