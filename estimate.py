#!/bin/python3
import numpy as np
import cv2
from sys import argv

scale = 15

fname = argv[1]
img = cv2.imread(fname)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).astype(np.float32)
gray = cv2.GaussianBlur(gray, (scale, scale), scale/5)
gray = np.abs(np.diff(np.diff(gray), axis=0)) / 10
threshold = np.quantile(gray, 0.999)
retval, labels, stats, centroids = cv2.connectedComponentsWithStats((gray > threshold).astype(np.uint8))
centroids = centroids[1:].astype(np.float32)

# for x, y in centroids:
    # cv2.circle(img, (round(x), round(y)), scale//3, (50, 50, 200))

print("shift by", centroids[1] - centroids[0])
shifted = centroids + (centroids[1] - centroids[0])
# for x, y in shifted:
    # cv2.circle(img, (round(x), round(y)), scale//3, (50, 200, 100))

flann = cv2.flann_Index()
flann.build(centroids, {'algorithm': 1, 'trees': 4})

indices, distances = flann.knnSearch(shifted, 1)
for (x1, y1), (index,), (distance,) in zip(shifted, indices, distances):
    if distance < scale**2:
        x2, y2 = centroids[index]
        cv2.line(img, (round(x1), round(y1)), (round(x2), round(y2)), (50, 50, 200))

if False:
    h, w = 8,8
    
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((h*w,3), np.float32)
    objp[:,:2] = np.mgrid[0:h,0:w].T.reshape(-1,2)
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (h, w), None)
    
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
    
        # termination criteria
        imgpoints.append(corners2)
    
        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (h, w), corners2, ret)

img[:-1,:-1,1] += (500*gray).astype(np.uint8)
cv2.imshow('img', img)
cv2.waitKey(0)
