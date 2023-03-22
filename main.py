import cv2 as cv

# Draw rectangles and text over specified coordinates
def draw_label(img, x, y, w, h, color, label):
    cv.rectangle(img, (x,y), (x+w,y+h), color, 2)
    cv.putText(img, label, (x+10,y+15), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Load the image in grayscale
image = cv.imread('images/stars.jpg')
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# Filter the image with a 5x5 gaussian kernel to remove possible noise
blurred = cv.GaussianBlur(gray, (5, 5), 0)

# Apply Otsu's threshold method (automatically determines the threshold value)
_, thr_img = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

# Generate the image's histogram
hist = cv.calcHist([thr_img], [0], None, [256], [0, 256])

# If there are more white pixels than black pixels, it means the background is white and objects are black, so colors are inverted
# because the findContours function assumes that objects in the image are lighter than the background by default
img_for_contours = thr_img
if hist[-1] > hist[0]:
    # Invert the colors of the image (detect white objects in a black background)
    img_for_contours = cv.bitwise_not(thr_img)

contours, hierarchy = cv.findContours(img_for_contours, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Store each area of contours
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
        # Check if the object is at least 2 times bigger than the others
        if cv.contourArea(cnt)/sorted(areas)[0] >= 2:
            categories["Undefined"] += 1
            indexes_borders.append(i)
            draw_label(output_img, x, y, w, h, (255, 0, 0), "Undefined")

# Remove contours and areas of objects that were already identified
list_contours = list(contours)
for index in sorted(indexes_borders, reverse=True):
    del list_contours[index]
new_contours = tuple(list_contours)

for index in sorted(indexes_borders, reverse=True):
    del areas[index]

areas.sort()

for i, cnt in enumerate(new_contours):
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

cv.imshow("Labeled objetcs", output_img)
cv.waitKey(0)
cv.destroyAllWindows()