import cv2
import numpy as np

# Load the image in grayscale
image = cv2.imread('gears.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
