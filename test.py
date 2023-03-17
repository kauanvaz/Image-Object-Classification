import cv2

# Read the image
img = cv2.imread('gears.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply a threshold to the grayscale image
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Find the contours in the binary image
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print(len(contours))
# Iterate through the contours and calculate their areas
for contour in contours:
    area = cv2.contourArea(contour)
    print(f"Contour area: {area}")
    cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)
    
# Display the image with the contours and their areas
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()