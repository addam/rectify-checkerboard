#!/bin/python3
import numpy as np
import cv2
from sys import argv

scale = 15

def find_corners(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = cv2.GaussianBlur(gray, (scale, scale), scale/5)
    gray = np.abs(np.diff(np.diff(gray), axis=0)) / 10
    threshold = np.quantile(gray, 0.999)
    mask = (gray > threshold).astype(np.uint8)
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    # ignore 0-th component because it is the background
    return centroids[1:].astype(np.float32)

def grid_match(points, directions=4):
    flann = cv2.flann_Index()
    flann.build(points, {'algorithm': 1, 'trees': 1})
    best_score = 0
    best_result = None
    for guess in range(len(points)):
        origin = points[guess:guess+1,:]
        # last row in the result array represents a "no match" point
        result = -np.ones((points.shape[0] + 1, directions), np.int32)
        score = 0
        for i, n in enumerate(flann.knnSearch(origin, directions+1)[0].T[1:]):
            if n == guess:
                score = -float("inf")
            target = points[n]
            shifted = points + (target - origin)
            indices, distances = flann.knnSearch(shifted, 1)
            indices[distances > scale**2] = -1
            result[indices[:,0], i] = np.arange(len(indices))
            score += sum(distances < scale**2)
        score -= sum(sum(result[:,i] == result[:,j]) for i in range(1,4) for j in range(i))
        if score > best_score:
            best_score, best_result = score, result
    return best_result[:-1]

def order_cross(indices):
    """indices: array(n, 4), each column are neighbor indices in a direction
    Ensures that the directions are ordered something like left, right, up, down
    <=> indices[:,0] is inverse permutation of indices[:,1]
    """
    n = indices
    current = sum(n[n[:,0],1] == np.arange(len(n))) + sum(n[n[:,2],3] == np.arange(len(n)))
    swapped = sum(n[n[:,0],2] == np.arange(len(n))) + sum(n[n[:,1],3] == np.arange(len(n)))
    if swapped > current:
        indices[:,1:3] = indices[:,2:0:-1]

def grid_positions(neighbors, directions):
    result = np.zeros((neighbors.shape[0], directions.shape[1]), np.int32)
    visited = np.zeros((neighbors.shape[0],), np.bool)
    while not visited.all():
        seed = next(i for i, v in enumerate(visited) if not v)
        boundary = [seed]
        while boundary:
            index = boundary.pop()
            visited[index] = True
            for n, step in zip(neighbors[index,:], directions):
                if n < 0:
                    continue
                elif not visited[n]:
                    result[n,:] = result[index] + step
                    boundary.append(n)
                else:
                    if not (result[n,:] == result[index] + step).all():
                        print(index, result[index], "->", n, result[index] + step, "!=", result[n,:])
    return result

fname = argv[1]
img = cv2.imread(fname)
points = find_corners(img)
directions = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
neighbors = grid_match(points, directions.shape[0])
order_cross(neighbors)
labels = grid_positions(neighbors, directions)
for (x, y), (i, j) in zip(points, labels):
    cv2.putText(img, f"{i}, {j}", (round(x), round(y)), cv2.FONT_HERSHEY_PLAIN, 1, (100, 150, 250))

# for x, y in centroids:
    # cv2.circle(img, (round(x), round(y)), scale//3, (50, 50, 200))

# for (x1, y1), (index,), (distance,) in zip(shifted, indices, distances):
    # if distance < scale**2:
        # x2, y2 = points[index]
        # cv2.line(img, (round(x1), round(y1)), (round(x2), round(y2)), (50, 50, 200))

# img[:-1,:-1,1] += (500*gray).astype(np.uint8)

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

cv2.imshow('img', img)
cv2.waitKey(0)
