import os
import sys
import cv2
import numpy as np

images_directory = sys.argv[1]
# Reading Images
images = os.listdir(images_directory)

for imagename in images:

    # reading image
    img = cv2.imread(f'{images_directory}/{imagename}')

    # creating a mask
    mask = np.zeros(img.shape[:2], np.uint8)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Getting edges
    edges = cv2.Canny(gray, 50, 200)
    # getting largest contour bounding Rectangle coordinates
    contours, hierarchy = cv2.findContours(
        edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt = contours[max_index]
    x, y, w, h = cv2.boundingRect(cnt)
    rect = (x, y, w, h)

    # using grabcut algorithm to segment the image
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = img*mask2[:, :, np.newaxis]

    # Saving resuts
    print(f"Saving {imagename} result")
    cv2.imwrite(f'contours results/{imagename}', img)
