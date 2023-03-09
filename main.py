import cv2 as cv
import numpy as np

# Load the image in grayscale
image = cv.imread('gears.jpg')
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# Filter the image with a 5x5 gaussian kernel to remove possible noise
blurred = cv.GaussianBlur(gray, (5, 5), 0)

# Apply Otsu's threshold method (automatically determines the threshold value)
ret, thresh = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

cv.imshow("Threshold", thresh)
cv.waitKey(0)