import cv2 as cv
import numpy as np

def draw_label(img, x, y, w, h, color, label):
    cv.rectangle(img, (x,y), (x+w,y+h), color, 2)
    cv.putText(img, label, (x,y-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Load the image in grayscale
image = cv.imread('images/gears.jpg')
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# Filter the image with a 5x5 gaussian kernel to remove possible noise
blurred = cv.GaussianBlur(gray, (5, 5), 0)

# Invert the colors of the image (detect white objects in a black background)
inverted_img = cv.bitwise_not(blurred)

# Apply Otsu's threshold method (automatically determines the threshold value)
_, thr_img = cv.threshold(inverted_img, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
#thr_img = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
contours, hierarchy = cv.findContours(thr_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
"""
cv.drawContours(image, contours, -1, (0, 0, 255), 2)

cv.imshow('Contours', image)
cv.waitKey(0)
cv.destroyAllWindows()
"""

areas = []
for cnt in contours:
    areas.append(cv.contourArea(cnt))

categories = {"Good":0, "Defective":0, "Undefined":0}
output_img = image.copy()
indexes_borders = []

for i, cnt in enumerate(contours):
    x,y,w,h = cv.boundingRect(cnt)
    # Check if the object is touching any edge
    if x == 0 or y == 0 or x+w == image.shape[1] or y+h == image.shape[0]:
        categories["Undefined"] += 1
        indexes_borders.append(i)
        draw_label(output_img, x, y, w, h, (255, 0, 0), "Undefined")
    else:
        if cv.contourArea(cnt)/sorted(areas)[0] >= 2:
            categories["Undefined"] += 1
            indexes_borders.append(i)
            draw_label(output_img, x, y, w, h, (255, 0, 0), "Undefined")

for index in sorted(indexes_borders, reverse=True):
    del contours[index]

for index in sorted(indexes_borders, reverse=True):
    del areas[index]

areas.sort()

for i, cnt in enumerate(contours):
    x,y,w,h = cv.boundingRect(cnt)
    ratio = round(cv.contourArea(cnt)/areas[-1], 2)
    # Check if the object is defective
    if ratio < 1.00:
        categories["Defective"] += 1
        draw_label(output_img, x, y, w, h, (0, 0, 255), "Defective")
    else:
        categories["Good"] += 1
        draw_label(output_img, x, y, w, h, (0, 255, 0), "Good")


# Display image and counts
print('Good: ', categories.get("Good"))
print('Defective: ', categories.get("Defective"))
print('Undefined: ', categories.get("Undefined"))

cv.imshow("Object Labels", output_img)
cv.waitKey(0)
cv.destroyAllWindows()