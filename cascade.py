import cv2
import sys
import numpy as np

# Stop Sign Cascade Classifier xml
stop_sign = cv2.CascadeClassifier('cascade_stop_sign.xml')

# Reading file

# reading image
img = cv2.imread(sys.argv[1])

# creating a mask
mask = np.zeros(img.shape[:2], np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
stop_sign_scaled = stop_sign.detectMultiScale(gray, 1.3, 5)

# Detect the stop sign, x,y = origin points, w = width, h = height
for (x, y, w, h) in stop_sign_scaled:
    rect = (x, y, w, h)

# using grapcut algorithm to segment the image
cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
img = img*mask2[:, :, np.newaxis]

# Saving resuts
print(f"Saving result as output.jpg")
cv2.imwrite('output.jpg', img)
